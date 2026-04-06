import json
import pickle
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
JSON_FILE = "apis.json"           # Your input file
INDEX_FILE = "apis.faiss"         # Output Vector Store
METADATA_FILE = "apis_metadata.pkl" # Output Metadata
MODEL_NAME = "all-MiniLM-L6-v2"   # Small, fast, effective model

def build_index():
    # 1. Check if file exists
    if not os.path.exists(JSON_FILE):
        print(f"❌ Error: {JSON_FILE} not found. Please create it first.")
        return

    # 2. Load API Data
    with open(JSON_FILE, 'r') as f:
        try:
            api_registry = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON: {e}")
            return

    print(f"🔹 Loaded {len(api_registry)} APIs from {JSON_FILE}")

    # 3. Prepare Text for Embedding
    # We want the vector search to find an API based on its description AND its filterable fields.
    descriptions = []
    metadata_list = []
    
    for api_key, config in api_registry.items():
        # Combine description + filter keys for better semantic matching
        # Example: "Fetch IT support tickets... (Fields: status, assignee, priority)"
        filter_keys = ", ".join(config.get("filter_mapping", {}).keys())
        text_to_embed = f"{config['description']} (Fields: {filter_keys})"
        
        descriptions.append(text_to_embed)
        metadata_list.append(api_key) # We store the key (e.g., 'tickets') to look up config later

    # 4. Initialize Model & Generate Embeddings
    print(f"🔹 Loading Model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("🔹 Generating Embeddings...")
    embeddings = model.encode(descriptions, convert_to_numpy=True)
    
    # FAISS requires float32
    embeddings = embeddings.astype('float32')

    # 5. Create & Train FAISS Index
    dimension = embeddings.shape[1] # 384 for all-MiniLM-L6-v2
    index = faiss.IndexFlatL2(dimension) # L2 Distance (Euclidean)
    index.add(embeddings)

    print(f"🔹 Built Index with {index.ntotal} vectors.")

    # 6. Save Artifacts
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump(metadata_list, f)

    print(f"✅ Success! Saved index to '{INDEX_FILE}' and metadata to '{METADATA_FILE}'")

if __name__ == "__main__":
    build_index()