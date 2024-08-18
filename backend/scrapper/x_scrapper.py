import asyncio
import logging
import re

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from backend.constant import USER_AGENT
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
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=USER_AGENT
            )
        self.page = await self.context.new_page()
        await self.page.goto(f"https://x.com/home")
        await self.page.click("//*[@id='react-root']/div/div/div[2]/main/div/div/div[1]/div[1]/div/div[3]/div[4]/a")
        await self.page.fill("//*[@id='layers']/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[4]/label/div/div[2]/div/input", X_EMAIL)
        await self.page.click('//*[@id="layers"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/button[2]')
        await self.page.fill('//*[@id="layers"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[1]/div/div[2]/label/div/div[2]/div/input', X_USERNAME)
        await self.page.click('//*[@id="layers"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[2]/div/div/div/button')
        await self.page.fill('//*[@id="layers"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[1]/div/div/div[3]/div/label/div/div[2]/div[1]/input', X_PASSWORD)
        await self.page.click('//*[@id="layers"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[2]/div/div[1]/div/div/button')
        logger.info(f"Sesión Iniciada en X con el email {X_EMAIL}")


    async def fetch_user_profile_html(self, username):
        await self.page.goto(f"https://x.com/{username}")
        await asyncio.sleep(2)
        # Esperar a que el contenido relevante esté cargado
        await self.page.wait_for_selector('//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div/section/div', state='attached')
        # Obtener el HTML de la sección deseada
        await asyncio.sleep(2)
        content_html = await self.page.inner_html('//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div/div/div')
        tweets_html = await self.page.inner_html('//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div/section/div')
        return content_html, tweets_html


    def process_user_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Verificar si el usuario está verificado
        verified_icon = soup.find('svg', {'aria-label': 'Verified account'})
        verified = verified_icon is not None
        
        # Buscar el nombre del usuario
        user_name_element = soup.find('div', {'class': 'css-146c3p1 r-bcqeeo r-1ttztb7 r-qvutc0 r-37j5jr r-adyw6z r-135wba7 r-1vr29t4 r-1awozwy r-6koalj r-1udh08x'})
        user_name = user_name_element.text if user_name_element else None

        # Buscar la fecha de unión
        join_date = soup.find('span', text=lambda text: text and 'Joined' in text)
        join_date = join_date.text if join_date else None
        
        # Información de seguidores, seguidos y suscripciones
        followers_info = {}
        labels = ["Following", "Followers", "Subscriptions", "Followed By"]
        followers_sections = soup.find_all('a', href=lambda href: href and any(x in href for x in ['following', 'followers', 'subscriptions']))
        for i, section in enumerate(followers_sections):
            numbers = section.find_all('span', class_='css-1jxf684')
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
    

    def extract_tweets(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        # Usar una expresión regular para encontrar la sección de los tweets
        tweets_section = soup.find('div', {'class': 'css-175oi2r'})
        # Extraer los primeros 5 tweets
        tweets = []
        if tweets_section:
        # Buscar cada tweet por su etiqueta específica y clase
            tweet_elements = tweets_section.find_all('div', {'class': 'css-175oi2r r-1igl3o0 r-qklmqi r-1adg3ll r-1ny4l3l'}, limit=5)
            for tweet in tweet_elements:
                tweet_text = tweet.get_text(strip=True)  # Extrae el texto de cada tweet
                tweets.append(tweet_text)        
        return tweets


    async def close(self):

        await self.browser.close()
        await self.playwright.stop()