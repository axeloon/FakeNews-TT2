from sqlalchemy.orm import Session
from .db_models import NoticiaRedesSociales
from datetime import date
from typing import List, Optional
import logging
from ..utils.response_models import SearchUserResponse

logger = logging.getLogger(__name__)

class NoticiaRedesSocialesUseCase:
    @staticmethod
    def create_noticia(db: Session, source: str, title: str, content: str, publication_date: date, author: str) -> Optional[NoticiaRedesSociales]:
        # Verificar si ya existe una noticia con el mismo contenido
        existing_noticia = db.query(NoticiaRedesSociales).filter(NoticiaRedesSociales.content == content).first()
        if existing_noticia:
            logger.info(f"Noticia con contenido duplicado no almacenada: {title}")
            return None

        db_noticia = NoticiaRedesSociales(
            source=source,
            title=title,
            content=content,
            publication_date=publication_date,
            author=author
        )
        db.add(db_noticia)
        db.commit()
        db.refresh(db_noticia)
        logger.info(f"Nueva noticia almacenada: {title}")
        return db_noticia

    @staticmethod
    def get_noticia(db: Session, noticia_id: int) -> Optional[NoticiaRedesSociales]:
        return db.query(NoticiaRedesSociales).filter(NoticiaRedesSociales.id == noticia_id).first()

    @staticmethod
    def get_all_noticias(db: Session, skip: int = 0, limit: int = 100) -> List[NoticiaRedesSociales]:
        return db.query(NoticiaRedesSociales).offset(skip).limit(limit).all()

    @staticmethod
    def delete_noticia(db: Session, noticia_id: int) -> bool:
        db_noticia = NoticiaRedesSocialesUseCase.get_noticia(db, noticia_id)
        if db_noticia:
            db.delete(db_noticia)
            db.commit()
            return True
        return False

    @staticmethod
    def store_search_result(db: Session, result: SearchUserResponse):
        stored_news = []
        try:
            for tweet in result.tweets:
                noticia = NoticiaRedesSocialesUseCase.create_noticia(
                    db=db,
                    source="Twitter",
                    title=f"Tweet by {tweet.usuario}",
                    content=tweet.texto,
                    publication_date=date.fromisoformat(tweet.fecha.split('T')[0]),
                    author=tweet.nombre
                )
                if noticia is not None:
                    stored_news.append(noticia)
            logger.info(f"Se han almacenado {len(stored_news)} noticias en la base de datos")
        except Exception as e:
            logger.error(f"Error almacenando noticias en la base de datos: {e}")