"""
LLM access for the benchmark: one backend for Gemini and local Ollama models,
with a response cache, call accounting, pacing and a hard budget.

Every response is cached on disk keyed by (provider, model, prompt hash,
generation settings), so re-running or resuming a run never repeats a call
that already succeeded. Failed calls are not cached.

The pipeline code in src/ expects a google-genai client. GenAIShim offers the
two methods it uses (models.generate_content and models.count_tokens) and
routes them through the backend, so the system under test runs unmodified.
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple


class BudgetExhausted(RuntimeError):
    """The run reached --max-live-calls."""


class QuotaExhausted(RuntimeError):
    """The provider refused the call for quota reasons (HTTP 429)."""


@dataclass
class CallRecord:
    purpose: str
    provider: str
    model: str
    cached: bool
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    latency_s: float = 0.0
    key: str = ""
    error: Optional[str] = None


@dataclass
class ModelSpec:
    provider: str
    model: str

    @classmethod
    def parse(cls, spec: str) -> "ModelSpec":
        if ":" not in spec:
            raise ValueError(f"model spec must look like 'gemini:MODEL' or 'ollama:MODEL', got {spec!r}")
        provider, model = spec.split(":", 1)
        if provider not in ("gemini", "ollama"):
            raise ValueError(f"unknown provider {provider!r}")
        return cls(provider, model)

    def __str__(self):
        return f"{self.provider}:{self.model}"


def _flatten(contents: Any) -> List[Tuple[str, Any]]:
    """google-genai style contents -> [("text", str) | ("image", (mime, bytes))]."""
    if contents is None:
        return []
    if isinstance(contents, str):
        return [("text", contents)]
    if isinstance(contents, (list, tuple)):
        out = []
        for item in contents:
            out.extend(_flatten(item))
        return out
    text = getattr(contents, "text", None)
    if text is not None:
        return [("text", text)]
    inline = getattr(contents, "inline_data", None)
    if inline is not None and getattr(inline, "data", None) is not None:
        return [("image", (inline.mime_type, inline.data))]
    parts = getattr(contents, "parts", None)
    if parts:
        return _flatten(list(parts))
    return [("text", str(contents))]


@dataclass
class LLMBackend:
    cache_dir: Path
    max_live_calls: Optional[int] = None
    min_interval_s: float = 0.0
    ollama_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 32768
    ollama_max_predict: int = 8192
    ledger_path: Optional[Path] = None
    gemini_attempts: int = 2  # tries per call when Gemini answers 503 (each try counts as a live call)
    records: List[CallRecord] = field(default_factory=list)
    live_calls: int = 0
    halted: Optional[BaseException] = None
    _last_live: float = 0.0
    _gemini_client: Any = None

    # -- public -----------------------------------------------------------
    def generate(self, spec: ModelSpec, contents: Any, *, temperature: Optional[float],
                 max_output_tokens: Optional[int], purpose: str) -> Tuple[str, CallRecord]:
        parts = _flatten(contents)
        key = self._key(spec, parts, temperature, max_output_tokens)
        cached = self._cache_get(spec, key)
        if cached is not None:
            rec = CallRecord(purpose, spec.provider, spec.model, True, key=key, **cached["usage"],
                             latency_s=cached["latency_s"])
            self.records.append(rec)
            return cached["text"], rec

        if self.halted is not None:
            raise self.halted
        if self.max_live_calls is not None and self.live_calls >= self.max_live_calls:
            self.halted = BudgetExhausted(f"live-call budget of {self.max_live_calls} reached")
            raise self.halted

        try:
            if spec.provider == "gemini":
                text, usage, latency = self._call_gemini(spec, contents, temperature, max_output_tokens)
            else:
                text, usage, latency = self._call_ollama(spec, parts, temperature, max_output_tokens)
        except Exception as exc:
            # The pipeline code swallows exceptions and carries on with empty results;
            # remember the failure so the runner stops instead of recording them.
            if self.halted is None:
                self.halted = exc
            raise
        rec = CallRecord(purpose, spec.provider, spec.model, False, key=key, latency_s=latency, **usage)
        self.records.append(rec)
        self._cache_put(spec, key, {"text": text, "usage": usage, "latency_s": latency,
                                    "provider": spec.provider, "model": spec.model,
                                    "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
        return text, rec

    def summary(self, prefix: str = "") -> dict:
        recs = [r for r in self.records if r.purpose.startswith(prefix)]
        return {
            "calls": len(recs),
            "live_calls": sum(not r.cached for r in recs),
            "cached_calls": sum(r.cached for r in recs),
            "input_tokens": sum(r.input_tokens for r in recs),
            "output_tokens": sum(r.output_tokens for r in recs),
            "thinking_tokens": sum(r.thinking_tokens for r in recs),
            "llm_latency_s": round(sum(r.latency_s for r in recs), 1),
        }

    # -- cache ------------------------------------------------------------
    def _key(self, spec, parts, temperature, max_output_tokens) -> str:
        canon = []
        for kind, value in parts:
            if kind == "text":
                canon.append({"text": value})
            else:
                canon.append({"image": value[0], "sha256": hashlib.sha256(value[1]).hexdigest()})
        payload = {"provider": spec.provider, "model": spec.model, "parts": canon,
                   "temperature": temperature, "max_output_tokens": max_output_tokens}
        if spec.provider == "ollama":
            payload["num_ctx"] = self.ollama_num_ctx
            payload["max_predict"] = self.ollama_max_predict
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _cache_path(self, spec, key) -> Path:
        safe_model = spec.model.replace("/", "_").replace(":", "_")
        return Path(self.cache_dir) / spec.provider / safe_model / key[:2] / f"{key}.json"

    def _cache_get(self, spec, key):
        path = self._cache_path(spec, key)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _cache_put(self, spec, key, value):
        path = self._cache_path(spec, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def _log_live(self, spec, status):
        if self.ledger_path:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{spec}\t{status}\n")

    def _pace(self):
        wait = self._last_live + self.min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_live = time.monotonic()

    # -- providers ----------------------------------------------------------
    def _call_gemini(self, spec, contents, temperature, max_output_tokens):
        from google import genai
        from google.genai import errors

        if self._gemini_client is None:
            key = os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("GOOGLE_API_KEY is not set")
            self._gemini_client = genai.Client(api_key=key)
        config = genai.types.GenerateContentConfig(temperature=temperature,
                                                   max_output_tokens=max_output_tokens)
        attempts = 0
        while True:
            if self.max_live_calls is not None and self.live_calls >= self.max_live_calls:
                self.halted = BudgetExhausted(f"live-call budget of {self.max_live_calls} reached")
                raise self.halted
            self._pace()
            self.live_calls += 1
            attempts += 1
            start = time.monotonic()
            try:
                response = self._gemini_client.models.generate_content(
                    model=spec.model, contents=contents, config=config)
            except errors.ClientError as exc:
                self._log_live(spec, f"client-error {exc.code}")
                if exc.code == 429:
                    self.halted = QuotaExhausted(str(exc)[:500])
                    raise self.halted
                raise
            except errors.ServerError as exc:
                self._log_live(spec, f"server-error {exc.code}")
                if attempts >= self.gemini_attempts:
                    raise
                time.sleep(20 * attempts)
                continue
            latency = time.monotonic() - start
            self._log_live(spec, "ok")
            usage = getattr(response, "usage_metadata", None)
            text = response.text or ""
            return text, {
                "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
                "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
                "thinking_tokens": int(getattr(usage, "thoughts_token_count", 0) or 0),
            }, round(latency, 2)

    def _call_ollama(self, spec, parts, temperature, max_output_tokens):
        # Text-only models: page images are dropped (llama3.2 has no vision input).
        text = "".join(value for kind, value in parts if kind == "text")
        predict = min(max_output_tokens or self.ollama_max_predict, self.ollama_max_predict)
        body = {
            "model": spec.model,
            "messages": [{"role": "user", "content": text}],
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.0 if temperature is None else temperature,
                        "num_ctx": self.ollama_num_ctx, "num_predict": predict, "seed": 0},
        }
        self.live_calls += 1
        start = time.monotonic()
        request = urllib.request.Request(f"{self.ollama_url}/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=3600) as resp:
                data = json.load(resp)
        except urllib.error.URLError as exc:
            self._log_live(spec, f"error {exc}")
            raise
        latency = time.monotonic() - start
        self._log_live(spec, "ok")
        return data["message"]["content"], {
            "input_tokens": int(data.get("prompt_eval_count") or 0),
            "output_tokens": int(data.get("eval_count") or 0),
            "thinking_tokens": 0,
        }, round(latency, 2)


class _Models:
    def __init__(self, shim):
        self._shim = shim

    def generate_content(self, model=None, contents=None, config=None):
        shim = self._shim
        text, rec = shim.backend.generate(
            shim.spec, contents,
            temperature=getattr(config, "temperature", None),
            max_output_tokens=getattr(config, "max_output_tokens", None),
            purpose=shim.purpose)
        usage = SimpleNamespace(prompt_token_count=rec.input_tokens,
                                candidates_token_count=rec.output_tokens,
                                thoughts_token_count=rec.thinking_tokens)
        return SimpleNamespace(text=text, usage_metadata=usage, candidates=[], prompt_feedback=None)

    def count_tokens(self, model=None, contents=None):
        chars = sum(len(v) for k, v in _flatten(contents) if k == "text")
        return SimpleNamespace(total_tokens=chars // 4)


class GenAIShim:
    """Stands in for google.genai.Client inside the pipeline code."""

    def __init__(self, backend: LLMBackend, spec: ModelSpec, purpose: str):
        self.backend, self.spec, self.purpose = backend, spec, purpose
        self.models = _Models(self)


def record_to_dict(rec: CallRecord) -> dict:
    d = asdict(rec)
    d.pop("key", None)
    return d
