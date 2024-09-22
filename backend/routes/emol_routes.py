from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from sqlalchemy.orm import Session
from typing import List
from ..database.connection import get_db
from ..scrapper.EmolScrapper import scrape_emol_historic
from ..database.use_cases import NoticiaRedesSocialesUseCase
from ..utils.response_models import EmolHistoricoResponse, EmolNoticiaResponse
from ..services.EmolService import EmolService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/emol/historico", response_model=EmolHistoricoResponse, tags=["EMOL"])
async def get_historic_emol(
    background_tasks: BackgroundTasks, 
    db: Session = Depends(get_db),
    id_fin: int = Query(1000000, description="ID final de la noticia"),
    cantidad_noticias: int = Query(20, description="Cantidad de noticias a scrappear")
):
    try:
        data = await scrape_emol_historic(id_fin, cantidad_noticias)
        
        if data:
            noticias_limpias = EmolService.limpiar_y_filtrar_noticias(data)
            background_tasks.add_task(NoticiaRedesSocialesUseCase.store_result, db, noticias_limpias)
        
        return EmolHistoricoResponse(noticias=[EmolNoticiaResponse(**noticia.dict()) for noticia in data], total=len(data))
    except Exception as e:
        logger.error(f"Error obteniendo noticias históricas de EMOL: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor al obtener noticias históricas de EMOL")