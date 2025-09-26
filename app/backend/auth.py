import os 
from datetime import datetime, timedelta
from typing import Optional
import logging

from fastapi import Depends, HTTPException, status 
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt 
from passlib.context import CryptContext
from sqlalchemy.orm import Session


from . import models, schemas
from .db import SessionLocal, get_db


logger = logging.getLogger(__name__)

SECRET_KEY = os.getenv("JWT_SECRET", "kjsnd1243")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode = {"exp": expire, "sub": subject}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def authenticate_user(db: Session, username: str, password: str) -> Optional[models.User]:
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> models.User:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        token_data = schemas.TokenPayload(**payload)
    except JWTError as exc:
        logger.warning("TOken invalid: %s", exc)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalid")
    if token_data.sub is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token doesn't contain a sub.")
    user = db.query(models.User).filter(models.User.username == token_data.sub).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
    return user



def require_admin(current_user: models.User = Depends(get_current_user)) -> models.User:
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not admin")
    return current_user


def seed_admin(db: Session) -> None:
    username = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
    password = os.getenv("DEFAULT_ADMIN_PASSWORD", "Admin123!")
    email = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@example.com")
    
    existing = db.query(models.User).filter(models.User.username == username).first()
    if existing:
        return 
    
    admin = models.User(
        username=username,
        email=email,
        hashed_password=hash_password(password),
        role='admin',
        is_active=True,
    )
    db.add(admin)
    db.commit()
    logger.info("Admin default has been seeded: %s", username)
    


def bootstrap_admin() -> None:
    with SessionLocal() as db:
        seed_admin(db)
        
