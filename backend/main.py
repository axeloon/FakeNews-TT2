from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Depends
from typing import List
from sqlalchemy.orm import Session
from .scrapper.XScrapper import XScrapper
from .services.FileService import FileService
from .constant import X_TWEET_IMAGE_PATH
from .utils.response_models import SearchUserResponse, NoticiaResponse
from .database.connection import get_db
from .database.use_cases import NoticiaRedesSocialesUseCase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

x_scrapper = XScrapper()

@app.on_event("startup")
async def startup_event():
    await x_scrapper.start()

@app.on_event("shutdown")
async def shutdown_event():
    await x_scrapper.close()

@app.get("/api/search_user/{username}", response_model=SearchUserResponse)
async def search_user(
    background_tasks: BackgroundTasks, 
    username: str, 
    tweets_len: int = Query(30, description="Cantidad de tweets a procesar"),
    db: Session = Depends(get_db)
):
    logger.info(f"Buscando usuario: {username}, para encontrar los {tweets_len} primeros tweets")
    try:
        result = await x_scrapper.search_user(username, tweet_limit=tweets_len)
        background_tasks.add_task(NoticiaRedesSocialesUseCase.store_search_result, db, result)
        background_tasks.add_task(FileService.clean_folders, [X_TWEET_IMAGE_PATH])
        return result
    except Exception as e:
        logger.error(f"Error al buscar usuario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/noticias", response_model=List[NoticiaResponse])
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