import os
from file_loader import load_file
from chunker import chunk_text
from embedding_generator import get_embeddings
from pinecone_handler import index, upsert_chunks, search_query
from deduplicator import (
    calculate_hash,
    load_dedup_index,
    is_already_processed,
    update_index
)

def process_all_files(folder_path):
    dedup_index = load_dedup_index()

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            print(f"\n📄 Processing: {filename}")
            content = load_file(file_path)

            if content and isinstance(content, str) and content.strip():
                content_hash = calculate_hash(content)

                if is_already_processed(filename, content_hash, dedup_index):
                    print(f"⏩ Skipping already processed file: {filename}")
                    continue

                chunks = chunk_text(content, chunk_size=300, overlap=50)
                embeddings = get_embeddings(chunks)

                print(f"✅ Extracted {len(chunks)} chunks | {len(embeddings)} embeddings")
                print("🧠 Sample Vector (first 5 dims):", embeddings[0][:5])

                upsert_chunks(index, filename, chunks, embeddings)
                update_index(filename, content_hash, dedup_index)
            else:
                print("⚠️ No content extracted.")

def process_single_file(file_path):
    if not os.path.isfile(file_path):
        print(f"❌ File not found: {file_path}")
        return

    content = load_file(file_path)
    if content and isinstance(content, str) and content.strip():
        dedup_index = load_dedup_index()
        content_hash = calculate_hash(content)
        filename = os.path.basename(file_path)

        if is_already_processed(filename, content_hash, dedup_index):
            print(f"⏩ Skipping already processed file: {filename}")
            return

        chunks = chunk_text(content, chunk_size=300, overlap=50)
        embeddings = get_embeddings(chunks)

        print(f"✅ Extracted {len(chunks)} chunks | {len(embeddings)} embeddings")
        print("🧠 Sample Vector (first 5 dims):", embeddings[0][:5])

        upsert_chunks(index, filename, chunks, embeddings)
        update_index(filename, content_hash, dedup_index)
        print(f"✅ File '{filename}' ingested successfully.")
    else:
        print("⚠️ No content extracted.")

if __name__ == "__main__":
    mode = input("Choose mode (ingest/query/single): ").strip().lower()

    if mode == "ingest":
        process_all_files("Files")
    elif mode == "query":
        user_input = input("\n🔍 Enter your query: ")
        search_query(index, user_input)
    elif mode == "single":
        file_path = input("📁 Enter full file path to ingest: ").strip()
        process_single_file(file_path)
    else:
        print("⚠️ Invalid mode. Please type 'ingest', 'query', or 'single'.")
