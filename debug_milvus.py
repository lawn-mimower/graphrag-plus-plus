
import logging
from src.milvus_ingestion import MilvusIngestionEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def query_milvus(query_text: str, top_k: int = 20, similarity_threshold: float = 0.5):
    """
    Initializes the Milvus Ingestion Engine and searches for a query.
    
    Args:
        query_text: The text to search for.
        top_k: The number of results to return.
        similarity_threshold: The minimum similarity score.
    """
    try:
        # Initialize the engine, it will connect to the existing Milvus DB
        engine = MilvusIngestionEngine(auto_connect=True)

        if not engine.milvus_client:
            logging.error("Failed to connect to Milvus. Ensure the database file exists and is not corrupted.")
            return

        logging.info(f"Successfully connected to Milvus collection: {engine.collection_name}")
        
        # Search for the entity
        logging.info(f"Searching for: '{query_text}' with top_k={top_k} and threshold={similarity_threshold}")
        results = engine.search_similar_entities(
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        # Print the results
        if results:
            print("\n--- Search Results ---")
            for result in results:
                print(
                    f"  Name: {result['canonical_name']}\n"
                    f"  Type: {result['entity_type']}\n"
                    f"  Score: {result['similarity_score']:.4f}\n"
                    f"  SQL ID: {result['sql_id']}\n"
                    f"  --------"
                )
        else:
            print("\n--- No results found matching the criteria. ---")

        # Close the connection
        engine.close()

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    # The entity you are searching for
    search_query = "director a"
    
    # Parameters for the search
    # Using a low threshold to catch any potential match
    search_top_k = 20
    search_threshold = 0.1

    print(f"Starting Milvus debug search for query: '{search_query}'")
    query_milvus(
        query_text=search_query,
        top_k=search_top_k,
        similarity_threshold=search_threshold
    )
