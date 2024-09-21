from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Date, Text
from .connection import Base

class NoticiaRedesSociales(Base):
    __tablename__ = "noticias_redes_sociales"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(50))
    title = Column(Text)
    content = Column(Text)
    publication_date = Column(Date)
    author = Column(String(100))
    created_at = Column(DateTime, default=lambda: datetime.now(datetime.UTC))
    updated_at = Column(DateTime, default=lambda: datetime.now(datetime.UTC), onupdate=lambda: datetime.now(datetime.UTC))