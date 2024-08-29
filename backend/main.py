from fastapi import FastAPI, HTTPException, Query
from .scrapper.x_scrapper import XScraper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

x_scraper = XScraper()

@app.on_event("startup")
async def startup_event():
    await x_scraper.start()

@app.on_event("shutdown")
async def shutdown_event():
    await x_scraper.close()

@app.get("/api/search_user/{username}")
async def search_user(username: str, tweets_len: int = Query(30, description="Cantidad de tweets a procesar")):
    logger.info(f"Buscando usuario: {username}, para encontrar los {tweets_len} primeros tweets")
    try:
        html_user_data, tweets_html = await x_scraper.fetch_user_profile_html(username)
        user_data = x_scraper.process_user_html(html_user_data)
        tweets = await x_scraper.extract_tweets(tweets_html, tweet_limit=tweets_len)
        logger.info(f"Usuario encontrado: {user_data}, Tweets: {len(tweets)}")
        return {
            "user_data": user_data,
            "tweets": tweets
        }
    except Exception as e:
        logger.error(f"Error al buscar usuario: {e}")
        raise HTTPException(status_code=500, detail=str(e))
