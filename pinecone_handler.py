import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
region = os.getenv("PINECONE_ENVIRONMENT")     # "us-west-4"
index_name = os.getenv("PINECONE_INDEX_NAME")  # "file-embed-index"

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # because MiniLM uses 384-dim vectors
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=region  # "us-west-4"
        )
    )

# Connect to index
index = pc.Index(index_name)

def upsert_chunks(index, file_id, chunks, embeddings):
    """
    Store chunks and their embeddings in Pinecone.

    Args:
        index: Pinecone index object
        file_id: str (e.g., "Test.txt")
        chunks: list of str
        embeddings: list of vectors
    """
    vectors_to_upsert = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{file_id}_{i}"
        metadata = {
            "source": file_id,
            "chunk_index": i,
            "text": chunk
        }
        vectors_to_upsert.append((vector_id, embedding.tolist(), metadata))

    index.upsert(vectors=vectors_to_upsert)
    print(f"📤 Upserted {len(vectors_to_upsert)} vectors to Pinecone.")

from embedding_generator import get_embeddings

def search_query(index, query, top_k=3):
    """
    Perform semantic search against Pinecone.

    Args:
        index: Pinecone index object
        query: str (the user input)
        top_k: number of results to return

    Returns:
        List of matches with score, text, and metadata
    """
    # Embed the query
    query_vector = get_embeddings([query])[0]  # Single vector

    # Run the query
    results = index.query(vector=query_vector.tolist(), top_k=top_k, include_metadata=True)

    matches = results["matches"] if "matches" in results else []
    for i, match in enumerate(matches):
        print(f"\n🔎 Match #{i+1}")
        print(f"Score: {match['score']:.4f}")
        print(f"Source: {match['metadata'].get('source')}")
        print(f"Text: {match['metadata'].get('text')}")
