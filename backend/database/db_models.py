import datetime
from sqlalchemy import Column, Integer, String, Date, DateTime, Text

from .connection import Base

class NoticiaRedesSociales(Base):
    __tablename__ = "noticias_redes_sociales"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(50))
    title = Column(Text)
    content = Column(Text)
    publication_date = Column(Date)
    author = Column(String(100))
    created_at = Column(DateTime, nullable=True, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)