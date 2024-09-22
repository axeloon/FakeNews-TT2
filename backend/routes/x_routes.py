from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from ..scrapper.XScrapper import XScrapper
from ..services.FileService import FileService
from ..constant import X_TWEET_IMAGE_PATH
from ..utils.response_models import SearchUserResponse
from ..database.connection import get_db
from ..database.use_cases import NoticiaRedesSocialesUseCase
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/search_user/{username}", response_model=SearchUserResponse)
async def search_user(
    background_tasks: BackgroundTasks, 
    username: str, 
    tweets_len: int = Query(30, description="Cantidad de tweets a procesar"),
    db: Session = Depends(get_db),
    x_scrapper: XScrapper = Depends(XScrapper.get_instance)
):
    logger.info(f"Buscando usuario: {username}, para encontrar los {tweets_len} primeros tweets")
    try:
        result = await x_scrapper.search_user(username, tweet_limit=tweets_len)
        background_tasks.add_task(NoticiaRedesSocialesUseCase.store_result, db, result)
        background_tasks.add_task(FileService.clean_folders, [X_TWEET_IMAGE_PATH])
        return result
    except Exception as e:
        logger.error(f"Error al buscar usuario: {e}")
        raise HTTPException(status_code=500, detail=str(e))