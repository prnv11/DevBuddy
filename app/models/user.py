from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, String, Text, TIMESTAMP, DateTime, text, func
from sqlalchemy.orm import relationship
from ..database import Base
from datetime import datetime, time
from typing import List
from .conversation import Conversation
 
class User(Base):
    __tablename__ = "user"

    id : Mapped[int] = mapped_column(Integer, primary_key=True)
    username : Mapped[str] = mapped_column(String(30), unique=True, nullable=False)
    email : Mapped[str] = mapped_column(String(254), unique=True, nullable=False)
    password : Mapped[str] = mapped_column(String(255), nullable=False)   # argon2 hashes are ~97 chars
    bio : Mapped[str | None] = mapped_column(Text, nullable=True)
    primary_tech_stack : Mapped[str | None] = mapped_column(String(100), nullable=True)   # set at onboarding
    learning_tech_stack : Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at : Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at : Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # conversations : Mapped[List["Conversation"]] 