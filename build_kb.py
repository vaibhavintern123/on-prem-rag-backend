# build_kb.py
import os
import re
import chromadb
from chromadb.utils import embedding_functions

# === CONFIG ===
TXT_FOLDER = "Knowledge_base"
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "txt_knowledge"

# --- Prefer GPU if available (optional) ---
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# === Init persistent ChromaDB with multilingual embeddings ===
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-large",
    device=DEVICE,
    normalize_embeddings=True,  # cosine similarity friendly
)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Create or load collection
try:
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    print("📂 Loaded existing collection:", COLLECTION_NAME)
except Exception:
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )
    print("🆕 Created new collection:", COLLECTION_NAME)


# === TXT text extraction ===
def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# === Sentence-aware chunking with overlap ===
def chunk_text(text, max_len=800, overlap=100):
    """
    Split text into chunks without breaking sentences.
    Adds overlap to preserve context at chunk boundaries.
    """
    # Split by sentence-ending punctuation (supports Hindi + English)
    sentences = re.split(r'(?<=[।.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_len:
            current_chunk += " " + sentence
        else:
            # Save current chunk
            chunks.append(current_chunk.strip())
            # Add overlap (last 100 chars)
            current_chunk = current_chunk[-overlap:] + " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# === Add / Refresh a single TXT file in DB ===
def add_txt_to_kb(file_path):
    doc_id = os.path.splitext(os.path.basename(file_path))[0]

    # Wipe any partial/old chunks for this file before re-adding
    existing_ids = []
    try:
        results = collection.get(where={"source": file_path})
        existing_ids = results["ids"]
    except Exception:
        pass

    if existing_ids:
        print(f"♻️  Removing old entries for '{file_path}'")
        collection.delete(ids=existing_ids)

    text = extract_text_from_txt(file_path)
    chunks = chunk_text(text, max_len=800, overlap=100)

    for idx, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"{doc_id}_{idx}"],
            metadatas=[{"source": file_path}]
        )

    print(f"✅ Added '{file_path}' to knowledge base with {len(chunks)} chunks.")


# === Ensure folder exists ===
if not os.path.exists(TXT_FOLDER):
    os.makedirs(TXT_FOLDER)
    print(f"📁 Created folder: {TXT_FOLDER}")

# === Process TXT files ===
for file in os.listdir(TXT_FOLDER):
    if file.lower().endswith(".txt"):
        txt_path = os.path.join(TXT_FOLDER, file)
        add_txt_to_kb(txt_path)

print("🎉 Knowledge base updated and saved to", CHROMA_DIR)
