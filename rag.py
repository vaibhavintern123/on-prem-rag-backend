import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
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

# --- Conversation memory ---
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

# === Generate assistant response ===
def generate_response(user_query: str):
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
        stream=False,  # Non-streaming
    )

    assistant_text = completion.choices[0].message.content

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
