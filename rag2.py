# === query_kb_api.py (FastAPI + CORS + Vertex AI Gemma + Chroma RAG) ===

import os
import json
import requests
import google.auth
import google.auth.transport.requests
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# === Request model ===
class ChatRequest(BaseModel):
    message: str

# === CONFIG ===
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "txt_knowledge"

# Vertex AI Endpoint
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "creds_data.json"
ENDPOINT_ID = "3108586553048301568"   # <-- replace with your endpoint id
PROJECT_ID = "947132053690"         # <-- replace with your project id
LOCATION = "us-central1"
DEDICATED_DOMAIN = f"3108586553048301568.us-central1-565476537267.prediction.vertexai.goog"

# --- Prefer GPU if available ---
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# === Load embedding function ===
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-large",
    device=DEVICE,
    normalize_embeddings=True
)

# === Load Chroma client & collection ===
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

# --- Conversation memory ---
conversation_history = [
    {"role": "system", "content": """You are a helpful assistant.
Answer the user queries using ONLY the provided context.
If the answer is not in the context, say politely you don’t know.
Provide clear explanations.
Remember previous turns in the conversation."""}
]

# === Auth helper ===
def get_access_token():
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_request = google.auth.transport.requests.Request()
    credentials.refresh(auth_request)
    return credentials.token

# === Smart n_results ===
def smart_n_results(user_query: str) -> int:
    words = len(user_query.split())
    if words <= 4:
        return 8
    elif words <= 10:
        return 6
    else:
        return 4

# === Retrieve context with citations ===
def retrieve_context(user_query: str):
    n = smart_n_results(user_query)
    results = collection.query(query_texts=[user_query], n_results=n)

    docs = results["documents"][0]
    ids = results["ids"][0]
    scores = results["distances"][0]

    threshold = 0.5
    filtered = [(doc, doc_id, score) for doc, doc_id, score in zip(docs, ids, scores) if score < threshold]

    if not filtered:
        filtered = list(zip(docs, ids, scores))

    seen = set()
    unique_docs, citations = [], []
    for doc, doc_id, score in filtered:
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)
            citations.append({"id": doc_id, "score": score, "snippet": doc[:len(doc)]})

    return "\n\n".join(unique_docs[:10]), citations

# === Ask Gemma (instead of Groq) ===
def ask_gemma(messages):
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
        # New Gemma chat schema
        return response["predictions"]["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        return f"Error: {json.dumps(response, indent=2)}"


# === Generate assistant response ===
def generate_response(user_query: str):
    context, citations = retrieve_context(user_query)

    # Merge context into the user message instead of adding another system role
    user_message = {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion:\n{user_query}"
    }

    # Messages: start with system + history + current user turn
    messages = conversation_history + [user_message]

    assistant_text = ask_gemma(messages)

    # Save to conversation history
    conversation_history.append({"role": "user", "content": user_query})
    conversation_history.append({"role": "assistant", "content": assistant_text})

    return {
        "text": assistant_text,
        "sources": citations
    }


# === FastAPI app ===
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Chat endpoint ===
@app.post("/chat")
async def chat(body: ChatRequest):
    try:
        response = generate_response(body.message)
        print(response)
        return JSONResponse(response)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# === Reset endpoint ===
@app.get("/reset")
async def reset():
    global conversation_history
    conversation_history = conversation_history[:1]
    return {"status": "ok", "message": "Conversation history cleared."}

# === Health check ===
@app.get("/")
async def health():
    return {"status": "ok"}
