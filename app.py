# main.py
import os
from datetime import datetime, timedelta
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, EmailStr
from bson import ObjectId
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# LangChain imports (used for embeddings / LLM)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ---------------- load environment ----------------
load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
if not MONGO_URL:
    raise RuntimeError("MONGO_URL must be set in .env")

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY must be set in .env")

ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
ALGORITHM = "HS256"

# OPTIONAL: If you want a global fallback key (not recommended for multi-user prod)
# set GLOBAL_OPENAI_API_KEY in .env and set FALLBACK_TO_GLOBAL_KEY = True
# FALLBACK_TO_GLOBAL_KEY = False
# GLOBAL_OPENAI_API_KEY = os.getenv("GLOBAL_OPENAI_API_KEY")

# ---------------- database ----------------
client = AsyncIOMotorClient(MONGO_URL)
db = client["user_auth_db"]
user_collection = db["users"]
embedding_collection = db["embeddings"]
chat_collection = db["chats"]

# ---------------- security ----------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="signin")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# ---------------- schemas ----------------
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class APIKeyModel(BaseModel):
    api_key: str


# ---------------- app ----------------
app = FastAPI(title="Multi-PDF Chat (FastAPI + MongoDB + LangChain)")

# CORS - restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend origin(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- auth endpoints ----------------
@app.post("/signup")
async def signup(user: UserCreate):
    existing = await user_collection.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_doc = {
        "username": user.username,
        "email": user.email,
        "password": hash_password(user.password),
        # "openai_api_key": None  # set later via /save_api_key
    }
    await user_collection.insert_one(user_doc)
    return {"message": "User registered successfully"}


@app.post("/signin", response_model=Token)
async def signin(payload: dict):
    email = payload.get("email")
    password = payload.get("password")
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")
    user = await user_collection.find_one({"email": email})
    if not user or not verify_password(password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user["email"]})
    return {"access_token": token, "token_type": "bearer"}


# ---------------- auth helper ----------------
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = await user_collection.find_one({"email": email})
    if user is None:
        raise credentials_exception
    return user


# ---------------- user API key endpoints ----------------
@app.post("/save_api_key")
async def save_api_key(data: APIKeyModel, current_user: dict = Depends(get_current_user)):
    key = (data.api_key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="Invalid API key")
    # NOTE: currently stored in plaintext; consider encrypting for production
    await user_collection.update_one({"_id": current_user["_id"]}, {"$set": {"openai_api_key": key}})
    return {"message": "API key saved successfully"}


@app.get("/get_api_key")
async def get_api_key(current_user: dict = Depends(get_current_user)):
    api_key = current_user.get("openai_api_key")
    # Return masked key and a boolean presence indicator
    if api_key:
        masked = api_key[:4] + "*" * max(0, len(api_key) - 8) + api_key[-4:] if len(api_key) > 8 else "****"
        return {"has_key": True, "masked": masked}
    return {"has_key": False, "masked": None}


# ---------------- upload pdf (create embeddings) ----------------
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import tempfile
import asyncio

import math
from bson import ObjectId
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiofiles
import tempfile

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """
    Upload a PDF safely, split embeddings into smaller MongoDB documents (<16MB each)
    """
    user_api_key = current_user.get("openai_api_key")
    if not user_api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key missing. Please add it in API Settings.")

    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"{datetime.utcnow().timestamp()}_{file.filename}")

    try:
        # ✅ Stream the file to disk
        async with aiofiles.open(tmp_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):  # 1MB at a time
                await out.write(chunk)

        # ✅ Load and split
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([p.page_content for p in pages])

        embeddings_client = OpenAIEmbeddings(model="text-embedding-3-small", api_key=user_api_key)
        texts = [c.page_content for c in chunks]

        # ✅ Async batch embedding to avoid rate limits
        async def embed_batch(batch):
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                return await loop.run_in_executor(pool, lambda: [embeddings_client.embed_query(t) for t in batch])

        batch_size = 50
        pdf_group_id = str(ObjectId())  # group ID for this PDF
        total_batches = math.ceil(len(texts) / batch_size)

        for i in range(total_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(texts))
            batch_texts = texts[start:end]
            batch_embeddings = await embed_batch(batch_texts)

            # Save each batch separately to prevent BSON > 16MB
            await embedding_collection.insert_one({
                "pdf_group_id": pdf_group_id,
                "user_id": str(current_user["_id"]),
                "filename": file.filename,
                "upload_time": datetime.utcnow(),
                "batch_index": i,
                "chunks": batch_texts,
                "embeddings": batch_embeddings,
            })

        return {"message": f"✅ Uploaded successfully in {total_batches} batches", "pdf_group_id": pdf_group_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------- list user pdfs ----------------
@app.get("/user_pdfs")
async def get_user_pdfs(current_user: dict = Depends(get_current_user)):
    """
    Return unique list of uploaded PDFs (grouped by pdf_group_id)
    """
    cursor = embedding_collection.find({"user_id": str(current_user["_id"])})
    pdfs_dict = {}

    async for doc in cursor:
        # Get group ID or fallback to legacy _id
        pdf_group_id = doc.get("pdf_group_id") or str(doc["_id"])
        filename = doc.get("filename")
        upload_time = doc.get("upload_time")

        # Keep the earliest upload time per PDF group
        if pdf_group_id not in pdfs_dict:
            pdfs_dict[pdf_group_id] = {
                "id": pdf_group_id,
                "filename": filename,
                "upload_time": upload_time
            }

    # Sort by upload time (newest first)
    pdfs = sorted(pdfs_dict.values(), key=lambda x: x["upload_time"], reverse=True)
    return {"pdfs": pdfs}



# ---------------- chat with selected pdf ----------------
@app.post("/chat")
async def chat_with_pdf(
    question: str = Form(...),
    pdf_id: str = Form(...),   # this is now pdf_group_id
    current_user: dict = Depends(get_current_user)
):
    """
    Steps:
      - Handle small-talk greetings instantly
      - Load all chunks + embeddings from pdf_group_id
      - Embed question using user's OpenAI API key
      - Retrieve top-k chunks (cosine similarity)
      - Call LLM (ChatOpenAI) using user's key
      - Save chat history
    """

    user_api_key = current_user.get("openai_api_key")
    if not user_api_key:
        raise HTTPException(
            status_code=400,
            detail="❌ OpenAI API key missing. Please add it in API Settings."
        )

    # 🗣️ Step 1: Handle small talk
    lower_q = question.lower().strip()
    casual_replies = {
        "hi": "Hey there! 👋 How can I help you today?",
        "hello": "Hello! 😊 Ask me anything about your document.",
        "hey": "Hey! What would you like to explore today?",
        "thanks": "You're very welcome! 🙏",
        "thank you": "Happy to help! 🤝",
        "ok": "Alright 👍",
        "good morning": "Good morning ☀️ Ready to learn something new?",
        "good night": "Good night 🌙 Sleep well!",
        "how are you": "I'm just lines of code, but I'm doing great! How about you?",
    }

    for key, reply in casual_replies.items():
        if key in lower_q:
            return {"question": question, "answer": reply}

    # 🧠 Step 2: Load all PDF chunks for this group
    cursor = embedding_collection.find({
        "pdf_group_id": pdf_id,
        "user_id": str(current_user["_id"])
    })

    chunks, embeddings = [], []
    async for doc in cursor:
        chunks.extend(doc.get("chunks", []))
        embeddings.extend(doc.get("embeddings", []))

    if not chunks or not embeddings:
        raise HTTPException(status_code=404, detail="No embeddings found for this PDF.")

    embeddings = np.array(embeddings)

    try:
        # 🔍 Step 3: Embed the user's question
        embeddings_client = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=user_api_key
        )
        query_emb = embeddings_client.embed_query(question)

        # 📏 Step 4: Compute cosine similarity
        sims = cosine_similarity([query_emb], embeddings)[0]
        top_k = 3
        idxs = np.argsort(sims)[-top_k:][::-1]
        context = "\n\n".join([chunks[i] for i in idxs])
        max_sim = float(np.max(sims)) if len(sims) else 0.0

        # 🧠 Step 5: Handle irrelevant questions (low similarity)
        if max_sim < 0.25:
            return {
                "question": question,
                "answer": "🤔 That question doesn’t seem related to your document. Try asking something based on its content!"
            }

        # 💬 Step 6: Query LLM
        llm = ChatOpenAI(
            api_key=user_api_key,
            model="gpt-4o-mini",
            temperature=0.2
        )

        prompt = PromptTemplate(
            template="""
You are a helpful assistant. Use ONLY the provided context to answer the user's question.
If the context is insufficient, say you don't know.

Context:
{context}

Question:
{question}
""",
            input_variables=["context", "question"]
        )

        final_prompt = prompt.format(context=context, question=question)
        llm_resp = llm.invoke(final_prompt)
        answer = getattr(llm_resp, "content", None) or str(llm_resp)

        # 💾 Step 7: Save chat history
        await chat_collection.insert_one({
            "user_id": str(current_user["_id"]),
            "pdf_group_id": pdf_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.utcnow()
        })

        return {"question": question, "answer": answer}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# ---------------- chat history ----------------
@app.get("/chat_history/{pdf_id}")
async def get_chat_history(pdf_id: str, current_user: dict = Depends(get_current_user)):
    cursor = chat_collection.find({"user_id": str(current_user["_id"]), "pdf_id": pdf_id}).sort("timestamp", -1)
    out = []
    async for d in cursor:
        out.append({"question": d.get("question"), "answer": d.get("answer"), "timestamp": d.get("timestamp")})
    return {"history": out}


# ---------------- root ----------------
@app.get("/")
def root():
    return {"message": "FastAPI Multi-PDF Chat Backend (per-user OpenAI keys) is running"}
