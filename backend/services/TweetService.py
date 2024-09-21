import asyncio
import logging
import requests
import os
from bs4 import BeautifulSoup
from backend import constant as c
from backend.utils.response_models import Tweet

logger = logging.getLogger(__name__)

class TweetService:
    @staticmethod
    def tweet_counter(html_content):
        soup = BeautifulSoup(html_content, c.HTML_PARSER)
        tweet_elements = soup.find_all(c.DIV, attrs={c.X_DATA_TESTID: c.X_TWEET_TEXT})
        return tweet_elements

    @staticmethod
    def extract_tweet(tweet):
        time_tag = tweet.find_previous(c.TIME)
        date = time_tag['datetime'] if time_tag else 'Fecha no disponible'
        
        user_name_tag = tweet.find_previous(c.DIV, attrs={c.X_DATA_TESTID: c.X_USER_NAME})
        full_user_name = user_name_tag.get_text(strip=True) if user_name_tag else 'Usuario no disponible'
        name, user_name = full_user_name.split('@')[0].strip(), '@' + full_user_name.split('@')[1].split('·')[0].strip() if '@' in full_user_name else None

        tweet_texts = tweet.find_all(c.SPAN)
        text_parts = [span.get_text(strip=True) for span in tweet_texts if 'media could not be played' not in span.get_text(strip=True)]
        tweet_text = ' '.join(text_parts) if text_parts else 'Contenido no disponible'
        
        interactions_div = tweet.find_next(c.DIV, {'role': 'group'})
        interactions = interactions_div['aria-label'] if interactions_div else 'Interacciones no disponibles'

        image_div = tweet.find_next(c.DIV, {c.X_DATA_TESTID: c.X_TWEET_IMAGE})
        image_url = image_div.find(c.IMG)['src'] if image_div and image_div.find(c.IMG) else None
        
        if image_url:
            response = requests.get(image_url)
            if not os.path.exists(c.X_TWEET_IMAGE_PATH):
                os.makedirs(c.X_TWEET_IMAGE_PATH)
            file_path = os.path.join(c.X_TWEET_IMAGE_PATH, f'{name}_{date.replace(":", "")}.jpg')
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            file_path = 'No image'
        
        return {
            'fecha': date,
            'nombre': name,
            'usuario': user_name,
            'texto': tweet_text,
            'interacciones': interactions,
            'imagen': file_path
        }

    @staticmethod
    async def load_more_tweets(page, tweets_html, tweet_limit):
        tweet_counter = TweetService.tweet_counter(tweets_html)
        total_tweets = len(tweet_counter)
        scroll_delay = 0.2
        max_scroll_depth = 10

        while total_tweets < tweet_limit:
            for i in range(max_scroll_depth):
                await page.evaluate(c.get_window_scroll_by(max_scroll_depth))
                await asyncio.sleep(scroll_delay)
            new_tweets_html = await page.inner_html(c.X_PROFILE_TWEETS_XPATH)
            if new_tweets_html not in tweets_html:
                tweets_html += new_tweets_html
                tweet_counter = TweetService.tweet_counter(tweets_html)
                total_tweets = len(tweet_counter)
                logger.info(f"Cantidad actual de tweets: {total_tweets}")
            await asyncio.sleep(2)
        await page.evaluate(c.WINDOW_SCROLL_TO)
        return tweets_html

    @staticmethod
    async def extract_tweets(page, html_content, tweet_limit):
        logger.info("Iniciando extracción de tweets.")
        tweet_elements = TweetService.tweet_counter(html_content)
        logger.info(f"Cantidad de tweets encontrados: {len(tweet_elements)}")
        tweets = []
        while len(tweets) < tweet_limit:
            for tweet in tweet_elements[:tweet_limit - len(tweets)]:
                tweet_data = TweetService.extract_tweet(tweet)
                tweets.append(Tweet(**tweet_data))
            if len(tweets) < tweet_limit:
                logger.info("Cargando más tweets.")
                html_content = await TweetService.load_more_tweets(page, html_content, tweet_limit)
                tweet_elements = TweetService.tweet_counter(html_content)
        logger.info("Extracción de tweets completada.")
        return tweets