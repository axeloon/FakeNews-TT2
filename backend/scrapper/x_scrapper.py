import asyncio
import logging
import requests
import os

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from backend import constant as c
from backend.config import X_EMAIL, X_USERNAME, X_PASSWORD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XScraper:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None


    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False)
        self.context = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=c.USER_AGENT
            )
        self.page = await self.context.new_page()
        await self.page.goto(f"{c.X_BASE_URL}/home")
        # Hacer clic en el botón de inicio de sesión
        await self.page.click(c.X_LOGIN_BUTTON_XPATH)
        await self.page.fill(c.X_EMAIL_INPUT_XPATH, X_EMAIL)
        await self.page.click(c.X_NEXT_BUTTON_XPATH)
        # Verificar si se requiere el nombre de usuario
        if await self.page.wait_for_selector(c.X_SELECTOR_MODAL_HEADER, state='visible'):
            await self.page.fill(c.X_USERNAME_INPUT_XPATH, X_USERNAME)
            await self.page.click(c.X_USERNAME_NEXT_BUTTON_XPATH)

        await self.page.fill(c.X_PASSWORD_INPUT_XPATH, X_PASSWORD)
        await self.page.click(c.X_LOGIN_SUBMIT_BUTTON_XPATH)
        logger.info(f"Sesión Iniciada en X con el email {X_EMAIL}")


    async def fetch_user_profile_html(self, username):
        await self.page.goto(f"{c.X_BASE_URL}/{username}")
        await asyncio.sleep(2)
        # Esperar a que el contenido relevante esté cargado
        await self.page.wait_for_selector(c.X_PROFILE_LOADING_SELECTOR, state='attached')
        # Obtener el HTML de la sección deseada
        await asyncio.sleep(2)
        content_html = await self.page.inner_html(c.X_PROFILE_CONTENT_XPATH)
        tweets_html = await self.page.inner_html(c.X_PROFILE_TWEETS_XPATH)
        await asyncio.sleep(2)
        return content_html, tweets_html


    def process_user_html(self, html_content):
        soup = BeautifulSoup(html_content, c.HTML_PARSER)
        
        # Verificar si el usuario está verificado
        verified_icon = soup.find('svg', {c.X_ARIA_LABEL_SELECTOR: c.X_VERIFIED_ICON_ARIA_LABEL})
        verified = verified_icon is not None
        
        # Buscar el nombre del usuario
        user_name_element = soup.find(c.DIV, {c.X_USER_NAME_DIV_CLASS})
        user_name = user_name_element.text if user_name_element else None

        # Buscar la fecha de unión
        join_date = soup.find(c.SPAN, text=lambda text: text and 'Joined' in text)
        join_date = join_date.text if join_date else None
        
        # Información de seguidores, seguidos y suscripciones
        followers_info = {}
        labels = c.X_PROFILE_LABELS
        followers_sections = soup.find_all('a', href=lambda href: href and any(x in href for x in c.X_PROFILE_LABELS))
        for i, section in enumerate(followers_sections):
            numbers = section.find_all('span', {c.X_FOLLOWERS_SPAN_CLASS})
            if numbers:
                value = numbers[0].text
                label = labels[i]
                followers_info[label] = value
        return {
            "user_name": user_name,
            "verified": verified,
            "join_date": join_date,
            "followers_info": followers_info
        }
    

    def tweet_counter(self, html_content):
        soup = BeautifulSoup(html_content, c.HTML_PARSER)
        # Buscar todos los elementos que contienen texto de tweet
        tweet_elements = soup.find_all(c.DIV, attrs={c.X_DATA_TESTID: c.X_TWEET_TEXT})
        return tweet_elements


    def extract_tweet(self, tweet):
        # Extraer la fecha, que generalmente se encuentra en el elemento <time> anterior al tweet
        time_tag = tweet.find_previous(c.TIME)
        date = time_tag['datetime'] if time_tag else 'Fecha no disponible'
        
        # Extraer el nombre de usuario y el handle
        user_name_tag = tweet.find_previous(c.DIV, attrs={c.X_DATA_TESTID: c.X_USER_NAME})
        full_user_name = user_name_tag.get_text(strip=True) if user_name_tag else 'Usuario no disponible'
        name, user_name = full_user_name.split('@')[0].strip(), '@' + full_user_name.split('@')[1].split('·')[0].strip() if '@' in full_user_name else None

        # Extraer el texto del tweet
        tweet_texts = tweet.find_all(c.SPAN)
        text_parts = [span.get_text(strip=True) for span in tweet_texts if 'media could not be played' not in span.get_text(strip=True)]
        tweet_text = ' '.join(text_parts) if text_parts else 'Contenido no disponible'
        
        # Extraer las interacciones, que se encuentran en un div con aria-label debajo del tweet
        interactions_div = tweet.find_next(c.DIV, {'role': 'group'})
        interactions = interactions_div['aria-label'] if interactions_div else 'Interacciones no disponibles'

        # Intentar obtener la URL de la imagen del tweet de un div específico
        image_div = tweet.find_next(c.DIV, {c.X_DATA_TESTID: c.X_TWEET_IMAGE})
        image_url = image_div.find(c.IMG)['src'] if image_div and image_div.find(c.IMG) else None
        
        # Si hay una imagen, descargar y guardar
        if image_url:
            response = requests.get(image_url)
            if not os.path.exists(c.X_TWEET_IMAGE_PATH):
                os.makedirs(c.X_TWEET_IMAGE_PATH)
            file_path = os.path.join(c.X_TWEET_IMAGE_PATH, f'{name}_{date.replace(':', '')}.jpg')
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

    
    async def load_more_tweets(self, tweets_html, tweet_limit):
        tweet_counter = self.tweet_counter(tweets_html)
        total_tweets = len(tweet_counter)
        scroll_delay = 0.2
        max_scroll_depth = 10

        while total_tweets < tweet_limit:
            for i in range(max_scroll_depth):
                await self.page.evaluate(c.get_window_scroll_by(max_scroll_depth))
                await asyncio.sleep(scroll_delay)
            new_tweets_html = await self.page.inner_html(c.X_PROFILE_TWEETS_XPATH)
            if new_tweets_html not in tweets_html:
                tweets_html += new_tweets_html
                tweet_counter = self.tweet_counter(tweets_html)
                total_tweets = len(tweet_counter)
                logger.info(f"Cantidad actual de tweets: {total_tweets}")
            await asyncio.sleep(2)
        await self.page.evaluate(c.WINDOW_SCROLL_TO)
        return tweets_html


    async def extract_tweets(self, html_content, tweet_limit):
        logger.info("Iniciando extracción de tweets.")
        tweet_elements = self.tweet_counter(html_content)
        logger.info(f"Cantidad de tweets encontrados: {len(tweet_elements)}")
        tweets = []
        while len(tweets) < tweet_limit:
            for tweet in tweet_elements[:tweet_limit - len(tweets)]:
                tweet_data = self.extract_tweet(tweet)
                tweets.append(tweet_data)
            if len(tweets) < tweet_limit:
                logger.info("Cargando más tweets.")
                html_content = await self.load_more_tweets(html_content, tweet_limit)
                tweet_elements = self.tweet_counter(html_content)
        logger.info("Extracción de tweets completada.")
        return tweets


    async def close(self):
        await self.browser.close()
        await self.playwright.stop()
        logger.info("Navegador cerrado y Playwright detenido.")