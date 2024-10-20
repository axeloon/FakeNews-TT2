from sqlalchemy.orm import Session
from .db_models import NoticiaRedesSociales
from datetime import date
from typing import List, Optional
import logging
from ..utils.response_models import SearchUserResponse, NoticiaBaseResponse

logger = logging.getLogger(__name__)

class NoticiaRedesSocialesUseCase:
    @staticmethod
    def create_noticia(db: Session, source: str, title: str, content: str, publication_date: date, author: str, is_false: bool ) -> Optional[NoticiaRedesSociales]:
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
            author=author,
            is_false=is_false
        )
        db.add(db_noticia)
        db.commit()
        db.refresh(db_noticia)
        logger.info(f"Nueva noticia almacenada: {title}")
        return db_noticia

    @staticmethod
    def get_noticia(db: Session, noticia_id: int) -> Optional[dict]:
        noticia = db.query(NoticiaRedesSociales).filter(NoticiaRedesSociales.id == noticia_id).first()
        return NoticiaRedesSocialesUseCase.noticia_to_dict(noticia) if noticia else None

    @staticmethod
    def get_all_noticias(db: Session, skip: int = 0, limit: int = 100) -> List[dict]:
        noticias = db.query(NoticiaRedesSociales).offset(skip).limit(limit).all()
        return [NoticiaRedesSocialesUseCase.noticia_to_dict(noticia) for noticia in noticias]

    @staticmethod
    def delete_noticia(db: Session, noticia_id: int) -> bool:
        db_noticia = NoticiaRedesSocialesUseCase.get_noticia(db, noticia_id)
        if db_noticia:
            db.delete(db_noticia)
            db.commit()
            return True
        return False

    @staticmethod
    def store_result(db: Session, result):
        stored_news = []
        try:
            if isinstance(result, SearchUserResponse):
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
            elif isinstance(result, list) and all(isinstance(item, NoticiaBaseResponse) for item in result):
                for noticia_data in result:
                    noticia = NoticiaRedesSocialesUseCase.create_noticia(
                        db=db,
                        source=noticia_data.source,
                        title=noticia_data.title,
                        content=noticia_data.content,
                        publication_date=noticia_data.publication_date,
                        author=noticia_data.author,
                        is_false=False
                    )
                    if noticia is not None:
                        stored_news.append(noticia)
            else:
                raise ValueError("Tipo de resultado no soportado")
            
            logger.info(f"Se han almacenado {len(stored_news)} noticias en la base de datos")
        except Exception as e:
            logger.error(f"Error almacenando noticias en la base de datos: {e}")

    @staticmethod
    def noticia_to_dict(noticia: NoticiaRedesSociales) -> dict:
        return {
            "id": noticia.id,
            "source": noticia.source,
            "title": noticia.title,
            "content": noticia.content,
            "publication_date": noticia.publication_date.isoformat(),
            "author": noticia.author,
            "created_at": noticia.created_at.isoformat(),
            "updated_at": noticia.updated_at.isoformat()
        }