from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from typing import List
from ..utils.response_models import NoticiaResponse
from ..database.connection import get_db
from ..database.use_cases import NoticiaRedesSocialesUseCase
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/noticias", response_model=List[NoticiaResponse])
async def get_noticias(
    skip: int = Query(0, description="Número de noticias a saltar"),
    limit: int = Query(100, description="Número máximo de noticias a retornar"),
    db: Session = Depends(get_db)
):
    try:
        noticias = NoticiaRedesSocialesUseCase.get_all_noticias(db, skip=skip, limit=limit)
        return [NoticiaResponse.model_validate(noticia) for noticia in noticias]
    except Exception as e:
        logger.error(f"Error al obtener noticias: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor al obtener noticias")