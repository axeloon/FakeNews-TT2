import datetime
from sqlalchemy import Column, Integer, String, Date, DateTime, Text, Boolean

from .connection import Base

class NoticiaRedesSociales(Base):
    __tablename__ = "noticias_redes_sociales"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(50), nullable=True)
    title = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    publication_date = Column(Date, nullable=True)
    author = Column(String(100), nullable=True)
    is_false = Column(Boolean, nullable=True, default=True)
    created_at = Column(DateTime, nullable=True, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)