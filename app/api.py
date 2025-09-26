import logging
import shutil
from pathlib import Path
from typing import List 
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.agent.qa_agent import answer_query
from app.models.classifier import LABELS, fit_and_save, load_model, predict
from app.utils.report import make_markdown_report

from app.backend import auth, models, schemas
from app.backend.db import Base, SessionLocal, engine, get_db
from app.backend.knowledge import KnowledgeStore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_ROOT = Path(__file__).resolve().parent
STORAGE_DIR  = APP_ROOT / "storage"
INDEX_DIR = STORAGE_DIR / "index"
UPLOADS_DIR = STORAGE_DIR / "uploads"

for path in (STORAGE_DIR, INDEX_DIR, UPLOADS_DIR):
    path.mkdir(parents=True, exist_ok=True)
    
knowledge = KnowledgeStore(index_dir=INDEX_DIR, uploads_dir=UPLOADS_DIR)


app = FastAPI(title="Sumbawa AI Legal Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    auth.bootstrap_admin()
    logger.info("Backend started.")
    
    
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/auth/login", response_model=schemas.TokenResponse)
def login(payload: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = auth.authenticate_user(db, payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    token = auth.create_access_token(user.username)
    return schemas.TokenResponse(access_token=token)


@app.post("/auth/register", response_model=schemas.UserBase, status_code=201)
def register_user(payload: schemas.UserRegister, db: Session = Depends(get_db)):
    for field, value in (("username", payload.username), ("email", payload.email)):
        if db.query(models.User).filter(getattr(models.User, field) == value).first():
            raise HTTPException(status_code=400, detail=f"{field} telah digunakan.")
    user = models.User(
        username=payload.username,
        email=payload.email,
        hashed_password=auth.hash_password(payload.password),
        role="user",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user 


@app.get("/auth/me", response_model=schemas.UserBase)
def read_me(current_user: models.User = Depends(auth.get_current_user)):
    return current_user

@app.post("/admin/users", response_model=schemas.UserBase)
def create_user(
    payload: schemas.UserCreate,
    db: Session = Depends(get_db),
    _: models.User = Depends(auth.require_admin),
):
    if payload.role not in {"admin", "user"}:
        raise HTTPException(status_code=400, detail="Invalid role")
    
    for field, value in (("username", payload.username), ("email", payload.email)):
        exists = db.query(models.User).filter(getattr(models.User, field) == value).first()
        if exists:
            raise HTTPException(status_code=400, detail=f"{field} already exists")
    
    user = models.User(
        username=payload.username,
        email=payload.email,
        hashed_password=auth.hash_password(payload.password),
        role=payload.role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user 


def _sanitize_filename(name: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    clean = clean.strip("._") or "upload"
    return clean


@app.post("/admin/upload", response_model=List[schemas.DocumentUploadResult])
async def upload_documents(
    files: List[UploadFile] =  File(...),
    current_user: models.User =  Depends(auth.require_admin),
    db: Session = Depends(get_db),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    results: List[schemas.DocumentUploadResult] = []
    for file in files:
        safe_name = _sanitize_filename(file.filename or "document")
        stored_name = f"{uuid4().hex}_{safe_name}"
        destination = UPLOADS_DIR / stored_name
        
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file.file.close()
        
        chunks_indexed = knowledge.add_file(file_path=destination, doc_id=file.filename or stored_name)
        document = models.Document(
            original_filename=file.filename or stored_name,
            stored_filename=stored_name,
            uploaded_by=current_user.id,
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        
        results.append(
            schemas.DocumentUploadResult(
                id=document.id,
                original_filename=document.original_filename,
                stored_filename=document.stored_filename,
                uploaded_at=document.uploaded_at,
                uploaded_by=document.uploaded_by,
                uploader_username=current_user.username,
                chunks_indexed=chunks_indexed,
            )
        )
    return results 


@app.get("/admin/documents", response_model=List[schemas.DocumentInfo])
def list_documents(
    _: models.User = Depends(auth.require_admin),
    db: Session = Depends(get_db),
):
    documents = db.query(models.Document).order_by(models.Document.uploaded_at.desc()).all()
    payload: List[schemas.DocumentInfo] = []
    for doc in documents:
        payload.append(
            schemas.DocumentInfo(
                id=doc.id,
                original_filename=doc.original_filename,
                stored_filename=doc.stored_filename,
                uploaded_at=doc.uploaded_at,
                uploaded_by=doc.uploaded_by,
                uploader_username=doc.uploader.username if doc.uploader else None,
            )
        )
    return payload


@app.post("/chat/ask", response_model=schemas.ChatResponse)
def ask_ai(
    payload: schemas.ChatRequest,
    _: models.User = Depends(auth.get_current_user),
):
    if knowledge.is_empty():
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Admin needs to upload documents.")
    
    hits = knowledge.search(payload.question, k=payload.top_k)
    if not hits:
        return schemas.ChatResponse(answer="No results found.", mode="empty", context=[])
    
    qa_result = answer_query(payload.question, hits, use_llm=payload.use_llm)
    
    classification = None 
    model = load_model()
    if model is None:
        texts = [hit["text"] for hit in hits]
        labels = [LABELS[i % len(LABELS)] for i in range(len(texts))]
        if texts:
            fit_and_save(texts, labels)
            model = load_model()
    
    if model:
        pred = predict(" ".join(hit["text"] for hit in hits[:3]), model)
        classification = schemas.ClassificationInfo(label=pred.label, score=pred.proba)
    
    context_payload = [
        schemas.ContextHit(
            doc_id=hit.get("doc_id"),
            source=hit.get("source"),
            text=hit.get("text", ""),
            score=hit.get("score"),
            page=hit.get("page"),
            section=hit.get("section"),
            section_chunk=hit.get("section_chunk"),
        )
        for hit in hits
    ]
    
    return schemas.ChatResponse(
        answer=qa_result.answer,
        mode=qa_result.mode,
        context=context_payload,
        classification=classification,
    )
    
    
    