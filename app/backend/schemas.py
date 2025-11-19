from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    

class LoginRequest(BaseModel):
    username: str 
    password: str 
    
class TokenPayload(BaseModel):
    sub: Optional[str] = None 
    exp: Optional[int] = None 
    
    
class UserBase(BaseModel):
    id: int 
    username: str 
    email: EmailStr
    role: str 
    is_active: bool 
    created_at: datetime
    
    class Config:
        orm_mode = True


class UserCreate(BaseModel):
    username: str 
    email: EmailStr
    password: str = Field(min_length=6)
    role: str = Field(default="user")
    

class DocumentMetadataInfo(BaseModel):
    source: str 
    data: Optional[Dict[str, Any]] = None


class DocumentInfo(BaseModel):
    id: int 
    original_filename: str 
    stored_filename: str 
    uploaded_at: datetime
    uploaded_by: int 
    uploader_username: Optional[str] = None 
    metadata: Optional[List[DocumentMetadataInfo]] = None 
    
    class Config:
        orm_mode = True 
        
    
class DocumentUploadResult(DocumentInfo):
    chunks_indexed: int 
    

class ChatRequest(BaseModel):
    question: str = Field(min_length=4)
    top_k: int = Field(default=5, ge=1, le=10)
    use_llm: bool = False 
    

class ContextHit(BaseModel):
    doc_id: Optional[str] = None 
    source: Optional[str] = None 
    text: str 
    score: Optional[float] = None 
    page: Optional[int] = None 
    section: Optional[int] = None 
    section_chunk: Optional[int] = None 
    structured_section: Optional[int] = None 
    bab: Optional[str] = None 
    bab_title: Optional[str] = None 
    bagian: Optional[str] = None 
    bagian_title: Optional[str] = None 
    paragraf: Optional[str] = None 
    paragraf_title: Optional[str] = None 
    pasal: Optional[str] = None 
    ayat: Optional[str] = None 
    
    

class ClassificationInfo(BaseModel):
    label: str 
    score: float 
    

class ChatResponse(BaseModel):
    answer: str 
    mode: str 
    context: List[ContextHit]
    classification: Optional[ClassificationInfo] = None 
    
    
class UserRegister(BaseModel):
    username: str 
    email: EmailStr
    password: str = Field(min_length=6)
    

class AnalysisRecordResponse(BaseModel):
    id: int 
    question: str 
    answer: str 
    mode: str 
    classification_label: Optional[str] = None 
    classification_score: Optional[float] = None 
    created_at: datetime 
    user_id: int 
    contexts: Optional[List[ContextHit]] = None 
    
    class Config:
        orm_mode = True 


class ScrapeRequest(BaseModel):
    keyword: str = Field(min_length=3)
    tentang: Optional[str] = Field(default=None)
    nomor: Optional[str] = Field(default=None)
    max_documents: int = Field(default=1, ge=1, le=10)
    downloads_per_document: int = Field(default=1, ge=1, le=5)
    auto_ingest: bool = Field(default=True)


class DownloadInfo(BaseModel):
    label: str
    url: str
    filename: str
    document_id: Optional[int] = None
    chunks_indexed: Optional[int] = None


class ScrapedDocument(BaseModel):
    title: str
    description: Optional[str] = None
    subjects: List[str] = Field(default_factory=list)
    detail_url: Optional[str] = None
    downloaded_files: List[DownloadInfo] = Field(default_factory=list)


class ScrapeResponse(BaseModel):
    keyword: str
    documents: List[ScrapedDocument]
    
