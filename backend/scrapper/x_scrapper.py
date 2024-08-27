import asyncio
import logging
import requests
import os

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
        # Hacer clic en el botón de inicio de sesión
        await self.page.click("//*[@id='react-root']/div/div/div[2]/main/div/div/div[1]/div[1]/div/div[3]/div[4]/a")
        await self.page.fill("//*[@id='layers']/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[4]/label/div/div[2]/div/input", X_EMAIL)
        await self.page.click('//*[@id="layers"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/button[2]')
        # Verificar si se requiere el nombre de usuario
        if await self.page.wait_for_selector('xpath=//*[@id="modal-header"]/span', state='visible'):
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
        await asyncio.sleep(2)
        # Scroll down lento para cargar más tweets
        scroll_delay = 0.2  # Tiempo de espera entre cada scroll pequeño
        max_scroll_depth = 10  # Cantidad de veces que se repetirá el scroll pequeño dentro de cada iteración grande
        for _ in range(5):  # Ajusta el rango según sea necesario para cargar más tweets
            for i in range(max_scroll_depth):
                # Scroll hacia abajo un poco a la vez
                await self.page.evaluate(f'window.scrollBy(0, window.innerHeight / {max_scroll_depth})')
                await asyncio.sleep(scroll_delay)  # Esperar para permitir que la página cargue más tweets
            new_tweets_html = await self.page.inner_html('//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/div/section/div')
            if new_tweets_html not in tweets_html:
                tweets_html += new_tweets_html
            await asyncio.sleep(2)  # Espera adicional después de cada bloque de scrolls
        
        # Asegurarse de llegar al final de la página después de todos los ciclos de scroll
        await self.page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
        await asyncio.sleep(2)  # Esperar para que se carguen los últimos tweets
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
    

    def extract_tweets(self, html_content, tweet_limit):
        soup = BeautifulSoup(html_content, 'html.parser')
        # Buscar todos los elementos que contienen texto de tweet
        tweet_elements = soup.find_all('div', attrs={'data-testid': 'tweetText'})
        tweets = []
        for tweet in tweet_elements[:tweet_limit]:
            # Extraer la fecha, que generalmente se encuentra en el elemento <time> anterior al tweet
            time_tag = tweet.find_previous('time')
            date = time_tag['datetime'] if time_tag else 'Fecha no disponible'
            
            # Extraer el nombre de usuario y el handle
            user_name_tag = tweet.find_previous('div', attrs={'data-testid': 'User-Name'})
            full_user_name = user_name_tag.get_text(strip=True) if user_name_tag else 'Usuario no disponible'
            name, user_name = full_user_name.split('@')[0].strip(), '@' + full_user_name.split('@')[1].split('·')[0].strip() if '@' in full_user_name else None

            # Extraer el texto del tweet
            tweet_texts = tweet.find_all('span')
            text_parts = [span.get_text(strip=True) for span in tweet_texts if 'media could not be played' not in span.get_text(strip=True)]
            tweet_text = ' '.join(text_parts) if text_parts else 'Contenido no disponible'
            
            # Extraer las interacciones, que se encuentran en un div con aria-label debajo del tweet
            interactions_div = tweet.find_next('div', {'role': 'group'})
            interactions = interactions_div['aria-label'] if interactions_div else 'Interacciones no disponibles'

            # Intentar obtener la URL de la imagen del tweet de un div específico
            image_div = tweet.find_next('div', {'data-testid': 'tweetPhoto'})
            image_url = image_div.find('img')['src'] if image_div and image_div.find('img') else None
            
            # Si hay una imagen, descargar y guardar
            if image_url:
                response = requests.get(image_url)
                folder_name = f"tweets"
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                file_path = os.path.join(folder_name, f'{name}_{date.replace(':', '')}.jpg')
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            
            # Agregar un diccionario con toda la información del tweet
            tweets.append({
                'fecha': date,
                'nombre': name,
                'usuario': user_name,
                'texto': tweet_text,
                'interacciones': interactions,
                'imagen': file_path if image_url else 'No image'
            })

        return tweets


    async def close(self):
        await self.browser.close()
        await self.playwright.stop()
        logger.info("Navegador cerrado y Playwright detenido.")