#!/usr/bin/env python3
"""
Entity Resolution Benchmarking Suite - Main Entry Point

Usage:
    python main.py [--dataset DATASET_PATH] [--config]
"""

import sys
import argparse
import logging
from pathlib import Path

from src.config import Config
from src.benchmark_harness import BenchmarkHarness


def setup_logging(log_level: str = None):
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    log_level = log_level or Config.LOG_LEVEL

    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Config.LOG_FILE)
        ]
    )


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Entity Resolution Benchmarking Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark on default dataset
  python main.py

  # Run on custom dataset
  python main.py --dataset /path/to/pdfs

  # Show configuration
  python main.py --config

  # Enable debug logging
  python main.py --log-level DEBUG
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help=f'Path to dataset directory (default: {Config.DATASET_DIR})'
    )

    parser.add_argument(
        '--config',
        action='store_true',
        help='Print configuration and exit'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=None,
        help='Logging level (default: LOG_LEVEL env var or INFO)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)

    # Print config if requested
    if args.config:
        Config.print_config()
        return 0

    # Print banner
    print("\n" + "=" * 80)
    print("  ENTITY RESOLUTION BENCHMARKING SUITE")
    print("  Comparing 5 Deduplication Methodologies")
    print("=" * 80 + "\n")

    # Validate dataset path
    dataset_path = Path(args.dataset) if args.dataset else Config.DATASET_DIR

    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        return 1

    if not dataset_path.is_dir():
        logger.error(f"Dataset path is not a directory: {dataset_path}")
        return 1

    # Check for supported documents (PDF, images, DOCX, XLSX)
    doc_files = [
        f for f in dataset_path.iterdir()
        if f.is_file() and f.suffix.lower() in Config.SUPPORTED_FORMATS
    ]
    if not doc_files:
        logger.error(
            f"No supported documents found in {dataset_path} "
            f"(supported: {', '.join(Config.SUPPORTED_FORMATS)})"
        )
        return 1

    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Found {len(doc_files)} document(s)")

    # Create and run benchmark harness
    try:
        harness = BenchmarkHarness()
        harness.run(dataset_path)

        print("\n✓ Benchmark completed successfully!")
        print(f"Results saved to: {Config.OUTPUTS_DIR}\n")

        return 0

    except KeyboardInterrupt:
        logger.warning("\nBenchmark interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
