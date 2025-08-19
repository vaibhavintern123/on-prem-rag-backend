# query_kb_api.py (FastAPI + CORS + Vertex AI Gemma + Chroma RAG with top-k chunk retrieval)

import os
import json
import requests
import google.auth
import google.auth.transport.requests
import chromadb
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from chromadb.utils import embedding_functions

# ===== CONFIG =====
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "pdf_knowledge"

# Vertex AI Endpoint
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds_data.json"
ENDPOINT_ID = "43991460227317760"
PROJECT_ID = "947132053690"
LOCATION = "europe-west4"
DEDICATED_DOMAIN = f"{ENDPOINT_ID}.{LOCATION}-{PROJECT_ID}.prediction.vertexai.goog"

# ===== INIT =====
app = FastAPI()
messages = []

# Enable CORS (allow all for dev; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Detect device for embeddings
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# Embedding function
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-large",
    device=DEVICE,
    normalize_embeddings=True,
)

# Chroma client + collection
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

# ===== Helper Functions =====
def get_access_token():
    """Get GCP access token from ADC."""
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_request = google.auth.transport.requests.Request()
    credentials.refresh(auth_request)
    return credentials.token

def rag_query(user_query: str, top_k: int = 3):
    """Retrieve top-k most relevant chunks (not full doc)."""
    results = collection.query(
        query_texts=[user_query],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    if not results["documents"]:
        return "", None

    chunks = results["documents"][0]
    sources = [meta.get("source") for meta in results["metadatas"][0]]

    context_text = "\n\n".join(chunks)
    return context_text, sources

def ask_gemma(full_prompt: str):
    """Send the prompt to Vertex AI Gemma chat model."""
    access_token = get_access_token()
    url = f"https://{DEDICATED_DOMAIN}/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict"

    payload = {
        "instances": [
            {
                "@requestFormat": "chatCompletions",
                "messages": messages,
                "max_tokens": 500
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=payload).json()
    try:
        return response["predictions"]["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return f"Error: {json.dumps(response, indent=2)}"

# ===== API Endpoint =====
@app.get("/ask")
def ask(user_input: str = Query(...)):
    context, pdf_sources = rag_query(user_input, top_k=3)
    print(f"🔍 RAG Context:\n{context}...")

    if context:
        source_info = ", ".join(set(s for s in pdf_sources if s))
        full_prompt = (
            f"You are a helpful assistant. The following are the top retrieved document chunks "
            f"from PDF(s) [{source_info}] relevant to the user's question.\n\n"
            f"Context:\n{context}\n\n"
            f"User Question: {user_input}\n"
            f"If the context does not contain the answer, reply politely that you don't know."
        )
    else:
        full_prompt = (
            f"You are a helpful assistant. No relevant PDF chunks were found.\n\n"
            f"User Question: {user_input}"
        )

    messages.append({"role": "user", "content": full_prompt})
    gemma_reply = ask_gemma(full_prompt)
    messages.append({"role": "assistant", "content": gemma_reply})

    return {"response": gemma_reply, "pdf_sources": pdf_sources}

# ===== Run =====
# uvicorn query_kb_api:app --reload
