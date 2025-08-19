# query_kb_api.py (FastAPI + CORS + Vertex AI Gemma + Chroma RAG)

import os
import json
import requests
import google.auth
import google.auth.transport.requests
import chromadb
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

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
messages = []  # conversation memory

# Enable CORS for all origins (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change ["*"] to your frontend domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)


# ===== Helper Functions =====
def get_access_token():
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    auth_request = google.auth.transport.requests.Request()
    credentials.refresh(auth_request)
    return credentials.token


def rag_query(user_query: str, n_results: int = 3):
    """Retrieve context from ChromaDB for the given query."""
    results = collection.query(query_texts=[user_query], n_results=n_results)
    context = "\n\n".join(results["documents"][0]) if results["documents"] else ""
    return context


def ask_gemma(full_prompt: str):
    """Send conversation (with RAG context) to Vertex AI Gemma."""
    access_token = get_access_token()
    url = f"https://{DEDICATED_DOMAIN}/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict"

    payload = {
        "instances": [
            {
                "@requestFormat": "chatCompletions",
                "messages": messages,
                "max_tokens": 1000
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=payload).json()

    try:
        model_reply = response["predictions"]["choices"][0]["message"]["content"]
        return model_reply.strip()
    except (KeyError, IndexError):
        return f"Error: {json.dumps(response, indent=2)}"


# ===== API Endpoint =====
@app.get("/ask")
def ask(user_input: str = Query(..., description="User's query to the KB")):
    # Retrieve RAG context
    context = rag_query(user_input)
    print(f"🔍 RAG Context:\n{context[:200]}...")  # Log first 200 chars for debugging
    if context:
        full_prompt = (
            f"You are a helpful assistant engaged in an ongoing conversation with the user. "
            f"Use the following context from the knowledge base to answer the user's latest question, "
            f"while considering previous messages in the conversation.\n\n"
            f"Context:\n{context}\n\n"
            f"User Question: {user_input}\n"
            f"If the context does not contain the answer, reply politely that you don't know, and feel free to ask clarifying questions."
        )
    else:
        full_prompt = (
            f"You are a helpful assistant engaged in a multi-turn conversation. "
            f"The knowledge base returned no relevant context for the user's latest question.\n\n"
            f"User Question: {user_input}\n"
            f"If you cannot answer, reply politely that you don't know, and feel free to ask for more details or clarification."
        )

    # Append user message with RAG context
    messages.append({"role": "user", "content": full_prompt})

    # Get model reply
    gemma_reply = ask_gemma(full_prompt)

    # Store assistant reply
    messages.append({"role": "assistant", "content": gemma_reply})

    return {"response": gemma_reply}


# ===== Run server =====
# uvicorn query_kb_api:app --reload
