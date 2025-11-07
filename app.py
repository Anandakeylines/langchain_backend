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
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """
    Upload a PDF -> extract text -> embed with user's OpenAI key -> save chunks+embeddings
    """
    # ensure user has API key
    user_api_key = current_user.get("openai_api_key")
    # If you want fallback to a global key (NOT recommended), uncomment and use GLOBAL_OPENAI_API_KEY:
    # if not user_api_key and FALLBACK_TO_GLOBAL_KEY:
    #     user_api_key = GLOBAL_OPENAI_API_KEY

    if not user_api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key missing. Please add it in API Settings.")

    # write temp file
    tmp_path = f"tmp_{datetime.utcnow().timestamp()}_{file.filename}"
    try:
        data = await file.read()
        with open(tmp_path, "wb") as fh:
            fh.write(data)

        # load and split
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        text = " ".join([p.page_content for p in pages])
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([text])

        # create embeddings with user's key
        embeddings_client = OpenAIEmbeddings(model="text-embedding-3-small", api_key=user_api_key)
        embedded_vectors = [embeddings_client.embed_query(c.page_content) for c in chunks]

        doc = {
            "user_id": str(current_user["_id"]),
            "filename": file.filename,
            "upload_time": datetime.utcnow(),
            "chunks": [c.page_content for c in chunks],
            "embeddings": embedded_vectors,
        }
        res = await embedding_collection.insert_one(doc)
        return {"message": "Embeddings saved successfully", "pdf_id": str(res.inserted_id)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# ---------------- list user pdfs ----------------
@app.get("/user_pdfs")
async def get_user_pdfs(current_user: dict = Depends(get_current_user)):
    cursor = embedding_collection.find({"user_id": str(current_user["_id"])})
    out = []
    async for d in cursor:
        out.append({"id": str(d["_id"]), "filename": d.get("filename"), "upload_time": d.get("upload_time")})
    return {"pdfs": out}


# ---------------- chat with selected pdf ----------------
@app.post("/chat")
async def chat_with_pdf(
    question: str = Form(...),
    pdf_id: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """
    User asks question -> if it's small talk, respond directly.
    Else, retrieve embeddings for selected PDF -> chat.
    """
    try:
        # 💬 1. Handle greetings and casual talk directly
        lower_q = question.lower().strip()
        common_phrases = {
            "hi": "Hey there! 👋 How can I help you today?",
            "hello": "Hello! 😊 Ask me anything about your document.",
            "hey": "Hey! What would you like to know?",
            "thanks": "You're very welcome! 🙏",
            "thank you": "Happy to help! 🤝",
            "ok": "Alright 👍",
            "good morning": "Good morning! ☀️",
            "good night": "Good night! 🌙",
        }
        for key, reply in common_phrases.items():
            if key in lower_q:
                return {"question": question, "answer": reply}

        # 🧩 2. Retrieve PDF embeddings as before
        pdf_data = await embedding_collection.find_one({
            "_id": ObjectId(pdf_id),
            "user_id": str(current_user["_id"])
        })
        if not pdf_data:
            raise HTTPException(status_code=404, detail="PDF not found for this user")

        chunks = pdf_data["chunks"]
        embeddings = np.array(pdf_data["embeddings"])

        # 🧠 3. Embed the question
        query_embedding = OpenAIEmbeddings(model="text-embedding-3-small").embed_query(question)
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[-3:][::-1]
        context = "\n\n".join([chunks[i] for i in top_indices])

        # 🧠 4. Generate intelligent answer
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        prompt = PromptTemplate(
            template="""You are a helpful assistant.
            Answer based ONLY on the context below. If unrelated, politely say you don’t know.

            Context:
            {context}

            Question:
            {question}
            """,
            input_variables=["context", "question"]
        )

        final_prompt = prompt.format(context=context, question=question)
        answer = llm.invoke(final_prompt).content.strip()

        # 💾 5. Save chat history
        await chat_collection.insert_one({
            "user_id": str(current_user["_id"]),
            "pdf_id": pdf_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.utcnow()
        })

        return {"question": question, "answer": answer or "Sorry, I couldn't find that in the document."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



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
