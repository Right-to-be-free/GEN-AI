import os
import hashlib
import json

DEDUP_FILE = "processed_files.json"

def calculate_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def load_dedup_index():
    if not os.path.exists(DEDUP_FILE):
        return {}
    with open(DEDUP_FILE, "r") as f:
        return json.load(f)

def save_dedup_index(index):
    with open(DEDUP_FILE, "w") as f:
        json.dump(index, f, indent=2)

def is_already_processed(filename, content_hash, index):
    return filename in index and index[filename] == content_hash

def update_index(filename, content_hash, index):
    index[filename] = content_hash
    save_dedup_index(index)
