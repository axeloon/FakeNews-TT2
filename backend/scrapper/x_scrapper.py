import asyncio
import logging

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
        self.browser = await self.playwright.chromium.launch(headless=False)
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


    async def get_tweets(self, username, max_results=10):
        url = f"https://twitter.com/{username}"
        await self.page.goto(url)
        # Implementa la lógica para extraer tweets aquí
        tweets = await self.page.query_selector_all("article")
        return tweets[:max_results]


    async def fetch_user_profile_html(self, username):
        await self.page.goto(f"https://x.com/{username}")
        await asyncio.sleep(2)
        # Esperar a que el contenido relevante esté cargado
        await self.page.wait_for_selector('//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div', state='attached')
        # Obtener el HTML de la sección deseada
        content_html = await self.page.inner_html('//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div/div/div')
        return content_html

    def process_user_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Verificar si el usuario está verificado
        verified_icon = soup.find('svg', {'aria-label': 'Verified account'})
        verified = verified_icon is not None
        
        # Buscar la fecha de unión
        join_date = soup.find('span', text=lambda text: text and 'Joined' in text)
        join_date = join_date.text if join_date else None
        
        # Información de seguidores, seguidos y suscripciones
        followers_info = {}
        followers_sections = soup.find_all('a', href=lambda href: href and any(x in href for x in ['following', 'followers', 'subscriptions']))
        for section in followers_sections:
            numbers = section.find_all('span', class_='css-1jxf684')
            if numbers:
                value = numbers[0].text
                label = numbers[1].text  # El texto que acompaña al número, debe ser Following, Followers o Subscriptions
                followers_info[label] = value

        return {
            "verified": verified,
            "join_date": join_date,
            "followers_info": followers_info
        }


    async def close(self):

        await self.browser.close()
        await self.playwright.stop()