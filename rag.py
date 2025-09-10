# rag_api.py
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
# === CONFIG ===
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "txt_knowledge"
GROQ_API_KEY = "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"  # 🔑 replace with env var in production

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

# === Init Groq client ===
client = Groq(api_key=GROQ_API_KEY)

# --- Conversation memory (shared per server instance) ---
conversation_history = [
    {"role": "system", "content": """You are a helpful assistant.
Answer the user queries using ONLY the provided context.
If the answer is not in the context, say whatever you make out of the context.
Provide clear explanations. 
Remember previous turns in the conversation."""}
]

# === Smart n_results ===
def smart_n_results(user_query: str) -> int:
    words = len(user_query.split())
    if words <= 4:
        return 8
    elif words <= 10:
        return 6
    else:
        return 4

# # === Retrieve & clean context ===
# def retrieve_context(user_query: str) -> str:
#     n = smart_n_results(user_query)
#     results = collection.query(query_texts=[user_query], n_results=n)

#     docs = results["documents"][0]
#     scores = results["distances"][0]

#     threshold = 0.5
#     filtered = [(doc, score) for doc, score in zip(docs, scores) if score < threshold]

#     if not filtered:
#         filtered = list(zip(docs, scores))

#     seen = set()
#     unique_docs = []
#     for doc, _ in filtered:
#         if doc not in seen:
#             seen.add(doc)
#             unique_docs.append(doc)

#     return "\n\n".join(unique_docs[:10])

# === FastAPI app ===
app = FastAPI()

# Enable CORS (allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Retrieve & clean context with citations ===
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
            citations.append({"id": doc_id, "score": score, "snippet": doc[:200]})  # snippet preview

    return "\n\n".join(unique_docs[:10]), citations


# === Stream generator (with citations) ===
async def generate_stream(user_query: str):
    context, citations = retrieve_context(user_query)
    context_message = {"role": "system", "content": f"Context for this turn:\n{context}"}

    messages = conversation_history + [context_message, {"role": "user", "content": user_query}]

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.7,
        max_completion_tokens=2000,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
    )

    full_response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response += token
            yield token
            await asyncio.sleep(0)

    # Append response + citations to conversation
    conversation_history.append({"role": "user", "content": user_query})
    conversation_history.append({"role": "assistant", "content": full_response})

    # Stream citations at the end
    yield "\n\n---\n**Sources:**\n"
    for c in citations:
        yield f"- {c['id']} (score={c['score']:.3f})\n"


# === API endpoint ===
@app.post("/chat")
async def chat(body: ChatRequest):
    user_message = body.message
    return StreamingResponse(generate_stream(user_message), media_type="text/markdown")


# === Reset endpoint ===
@app.get("/reset")
async def reset():
    global conversation_history
    conversation_history = conversation_history[:1]
    return {"status": "ok", "message": "Conversation history cleared."}
