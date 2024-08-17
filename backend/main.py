from fastapi import FastAPI, HTTPException
from .config import X_API_KEY, X_API_SECRET, X_BEARER_TOKEN
from .scrapper.x_scrapper import XScraper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


x_scraper = XScraper()

@app.get("/api/search_user/{username}")
def search_user(username: str):
    logger.info(f"Buscando usuario: {username}")
    try:
        user_data = x_scraper.search_user(username)
        logger.info(f"Usuario encontrado: {user_data}")
        return user_data
    except Exception as e:
        logger.error(f"Error al buscar usuario: {e}")
        raise HTTPException(status_code=500, detail="Error al buscar usuario")


@app.get("/api/hola_mundo")
def hola_mundo():
    logger.info(f"X_API_KEY: {X_API_KEY}")
    logger.info(f"X_API_SECRET: {X_API_SECRET}")
    logger.info(f"X_BEARER_TOKEN: {X_BEARER_TOKEN}")
    return {"message": "Hola mundo"}