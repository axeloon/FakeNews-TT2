import asyncio
import logging
import requests
import os

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from backend import constant as c
from backend.config import X_EMAIL, X_USERNAME, X_PASSWORD
from backend.services.TweetService import TweetService
from backend.models import UserData, SearchUserResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XScrapper:
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
        try:
            await self.page.wait_for_selector(c.X_SELECTOR_MODAL_HEADER, state='visible', timeout=1000)
            await self.page.fill(c.X_USERNAME_INPUT_XPATH, X_USERNAME)
            await self.page.click(c.X_USERNAME_NEXT_BUTTON_XPATH)
        except:
            logger.info("No se requiere el nombre de usuario, continuando con el ingreso de la contraseña.")
        
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
        return UserData(
            user_name=user_name,
            verified=verified,
            join_date=join_date,
            followers_info=followers_info
        )


    async def search_user(self, username, tweet_limit=10):
        content_html, tweets_html = await self.fetch_user_profile_html(username)
        user_data = self.process_user_html(content_html)
        tweets = await TweetService.extract_tweets(self.page, tweets_html, tweet_limit)
        return SearchUserResponse(user_data=user_data, tweets=tweets)

    async def close(self):
        await self.browser.close()
        await self.playwright.stop()
        logger.info("Navegador cerrado y Playwright detenido.")