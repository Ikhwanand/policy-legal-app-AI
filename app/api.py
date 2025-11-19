import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List 
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.agent.qa_agent import answer_query
from app.models.classifier import ensure_default_model, load_model, predict
from app.nlp.peraturan_scraper import download_file as scraper_download_file, search_peraturan

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
SCRAPER_ALLOWED_EXTENSIONS = {".pdf", ".docx"}


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
        db.flush()
        db.add(
            models.DocumentMetadata(
                document_id=document.id,
                source="manual",
                data={
                    "ingest_type": "upload",
                    "original_filename": file.filename,
                },
            )
        )
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


@app.post("/admin/scrape", response_model=schemas.ScrapeResponse)
def scrape_peraturan(
    payload: schemas.ScrapeRequest,
    current_user: models.User = Depends(auth.require_admin),
    db: Session = Depends(get_db),
):
    try:
        search_results = search_peraturan(
            payload.keyword,
            limit=payload.max_documents,
            tentang=payload.tentang,
            nomor=payload.nomor,
        )
    except Exception as exc:  # pragma: no cover - HTTP errors
        logger.exception("Failed to scrape peraturan.bpk.go.id: %s", exc)
        raise HTTPException(status_code=502, detail=f"Gagal mengakses peraturan.bpk.go.id: {exc}") from exc
    
    documents_payload: List[schemas.ScrapedDocument] = []
    for result in search_results:
        download_infos: List[Dict] = []
        for download in result.downloads[: payload.downloads_per_document]:
            info_payload: Dict[str, Any] = {
                "label": download.label,
                "url": download.url,
                "filename": download.filename,
            }
            if payload.auto_ingest:
                suffix = Path(download.filename).suffix.lower()
                if suffix and suffix not in SCRAPER_ALLOWED_EXTENSIONS:
                    logger.info("Skip %s (unsupported extension %s)", download.filename, suffix)
                    continue
                stored_name = f"{uuid4().hex}_{_sanitize_filename(download.filename)}"
                destination = UPLOADS_DIR / stored_name
                try:
                    scraper_download_file(download.url, destination)
                except Exception as exc:  # pragma: no cover - network errors
                    logger.warning("Failed to download %s: %s", download.url, exc)
                    continue
                chunks_indexed = knowledge.add_file(file_path=destination, doc_id=result.title or download.filename)
                document = models.Document(
                    original_filename=download.filename or result.title or stored_name,
                    stored_filename=stored_name,
                    uploaded_by=current_user.id,
                )
                db.add(document)
                db.flush()
                db.add(
                    models.DocumentMetadata(
                        document_id=document.id,
                        source="bpk_scraper",
                        data={
                            "keyword": payload.keyword,
                            "title": result.title,
                            "detail_url": result.detail_url,
                            "subjects": result.subjects,
                            "download_label": download.label,
                            "download_url": download.url,
                        },
                    )
                )
                db.commit()
                db.refresh(document)
                info_payload["document_id"] = document.id
                info_payload["chunks_indexed"] = chunks_indexed
            download_infos.append(info_payload)
        documents_payload.append(
            schemas.ScrapedDocument(
                title=result.title,
                description=result.description,
                subjects=result.subjects,
                detail_url=result.detail_url,
                downloaded_files=download_infos,
            )
        )
    return schemas.ScrapeResponse(keyword=payload.keyword, documents=documents_payload)


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
                metadata=[
                    schemas.DocumentMetadataInfo(source=meta.source, data=meta.data)
                    for meta in doc.metadata_entries
                ],
            )
        )
    return payload


@app.post("/chat/ask", response_model=schemas.ChatResponse)
def ask_ai(
    payload: schemas.ChatRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    if knowledge.is_empty():
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Admin needs to upload documents.")
    
    hits = knowledge.search(payload.question, k=payload.top_k)
    if not hits:
        return schemas.ChatResponse(answer="No results found.", mode="empty", context=[])
    
    qa_result = answer_query(payload.question, hits, use_llm=payload.use_llm)
    
    classification = None 
    model = load_model() or ensure_default_model()
    
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
            structured_section=hit.get("structured_section"),
            bab=hit.get("bab"),
            bab_title=hit.get("bab_title"),
            bagian=hit.get("bagian"),
            bagian_title=hit.get("bagian_title"),
            paragraf=hit.get("paragraf"),
            paragraf_title=hit.get("paragraf_title"),
            pasal=hit.get("pasal"),
            ayat=hit.get("ayat"),
        )
        for hit in hits
    ]
    
    response_payload = schemas.ChatResponse(
        answer=qa_result.answer,
        mode=qa_result.mode,
        context=context_payload,
        classification=classification,
    )
    
    analysis_record = models.AnalysisRecord(
        question=payload.question,
        answer=qa_result.answer,
        mode=qa_result.mode,
        classification_label=classification.label if classification else None,
        classification_score=classification.score if classification else None,
        user_id=current_user.id,
        contexts=jsonable_encoder(context_payload),
    )
    db.add(analysis_record)
    db.commit()
    
    return response_payload
    
    
@app.get("/analyses/history", response_model=List[schemas.AnalysisRecordResponse])
def list_analysis_history(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50,
):
    query = db.query(models.AnalysisRecord).order_by(models.AnalysisRecord.created_at.desc())
    if current_user.role != "admin":
        query = query.filter(models.AnalysisRecord.user_id == current_user.id)
    records = query.limit(limit).all()
    payload: List[schemas.AnalysisRecordResponse] = []
    for record in records:
        contexts = record.contexts or []
        payload.append(
            schemas.AnalysisRecordResponse(
                id=record.id,
                question=record.question,
                answer=record.answer,
                mode=record.mode,
                classification_label=record.classification_label,
                classification_score=record.classification_score,
                created_at=record.created_at,
                user_id=record.user_id,
                contexts=[schemas.ContextHit(**ctx) for ctx in contexts],
            )
        )
    return payload
    
    
    
