from datetime import datetime
from typing import List, Optional

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
    

class DocumentInfo(BaseModel):
    id: int 
    original_filename: str 
    stored_filename: str 
    uploaded_at: datetime
    uploaded_by: int 
    uploader_username: Optional[str] = None 
    
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
    
    
