from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from ..scrapper.XScrapper import XScrapper
from ..services.FileService import FileService
from ..constant import X_TWEET_IMAGE_PATH
from ..utils.response_models import SearchUserResponse
from ..database.connection import get_db
from ..database.use_cases import NoticiaRedesSocialesUseCase
import logging
import json
from pathlib import Path

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

@router.get("/search_users_list")
async def search_users_list(
    background_tasks: BackgroundTasks,
    list_type: str = Query(..., description="Tipo de lista a procesar (whitelist o blacklist)"),
    tweets_len: int = Query(15, description="Cantidad de tweets a procesar"),
    db: Session = Depends(get_db),
    x_scrapper: XScrapper = Depends(XScrapper.get_instance)
):
    logger.info(f"Iniciando búsqueda de usuarios en {list_type}")
    try:
        # Validar tipo de lista
        if list_type not in ['whitelist', 'blacklist']:
            raise HTTPException(status_code=400, detail="El tipo de lista debe ser 'whitelist' o 'blacklist'")
            
        # Leer archivo según tipo
        file_path = Path(__file__).parent.parent / "data" / f"{list_type}_x.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            users_list = json.load(file)
        
        results = []
        all_stored_news = []
        for user in users_list:
            try:
                logger.info(f"Procesando usuario: {user['usuario']}")
                result = await x_scrapper.search_user(user['usuario'], tweet_limit=tweets_len)
                results.append(result)
                # Almacenar resultados inmediatamente
                stored_news = NoticiaRedesSocialesUseCase.store_result(db, result)
                all_stored_news.extend(stored_news)
            except Exception as e:
                logger.error(f"Error procesando usuario {user['usuario']}: {e}")
                continue
        
        # Limpiar carpetas al finalizar
        background_tasks.add_task(FileService.clean_folders, [X_TWEET_IMAGE_PATH])
        
        return {"message": f"Procesados {len(results)} usuarios de {list_type} exitosamente, se almacenaron {len(all_stored_news)} noticias en la base de datos"}
    except Exception as e:
        logger.error(f"Error en proceso de {list_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
