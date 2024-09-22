import requests
import json
import logging
from datetime import datetime
from bs4 import BeautifulSoup
from backend import constant
from backend.utils.response_models import EmolNoticiaResponse, NoticiaBaseResponse

logger = logging.getLogger(__name__)

class EmolService:
    @staticmethod
    def _cargar_temas_interes():
        try:
            with open('backend/data/temas_intereses.json', 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Error al cargar temas_intereses.json: {e}")
            return {}

    @staticmethod
    async def obtener_noticia(id_noticia):
        url = f"{constant.NEWSAPI_NOTICIA}{id_noticia}"
        headers = {'User-Agent': constant.USER_AGENT}
        logger.info(f"URL: {url}")
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"No se pudieron encontrar noticias con el ID {id_noticia}, status code: {response.status_code}")
            return None

    @staticmethod
    def es_noticia_de_interes(source):
        temas_interes = EmolService._cargar_temas_interes()
        temas_noticia = {str(tema['id']) for tema in source.get('temas', [])}
        return bool(set(temas_interes.keys()) & temas_noticia)

    @staticmethod
    def limpiar_html(contenido_html):
        soup = BeautifulSoup(contenido_html, "html.parser")
        texto_limpio = soup.get_text(separator=" ", strip=True)
        return texto_limpio

    @staticmethod
    def procesar_noticia(source):
        news_id = source['id']
        category = source['seccion']
        title = source['titulo']
        subtitle = source.get('bajada', [{'texto': ''}])[0]['texto']
        body = EmolService.limpiar_html(source['texto'])

        if constant.TABLE in body or constant.TABLA_EN_NOTICIA in body:
            logger.info(f"Noticia ID {news_id} omitida por contener tablas HTML.")
            return None

        publication_datetime = datetime.fromisoformat(source['fechaPublicacion'])
        publication_date = publication_datetime.strftime('%Y-%m-%d')
        publication_time_str = source['fechaPublicacion'].split('T')[1][:5]

        return EmolNoticiaResponse(
            id=str(news_id),
            url=source['permalink'],
            title=title,
            subtitle=subtitle,
            content=body,
            publication_date=publication_date,
            publication_time=publication_time_str,
            category=category,
            author=source['autor'],
            publisher="EMOL",
            headline="historic"
        )

    @staticmethod
    def limpiar_y_filtrar_noticias(noticias):
        noticias_limpias = []
        for noticia in noticias:
                noticia_limpia = EmolService._limpiar_noticia(noticia)
                noticias_limpias.append(noticia_limpia)
        return noticias_limpias

    @staticmethod
    def _limpiar_noticia(noticia):
        content_limpio = EmolService._eliminar_enlaces_relacionados(noticia.content)
        
        return NoticiaBaseResponse(
            source="EMOL",
            title=noticia.title,
            content=content_limpio,
            publication_date=datetime.strptime(noticia.publication_date, '%Y-%m-%d').date(),
            author=noticia.author
        )

    @staticmethod
    def _eliminar_enlaces_relacionados(contenido):
        lineas = contenido.split('\n')
        lineas_limpias = [linea for linea in lineas if not linea.strip().startswith('{RELACIONADA')]
        return '\n'.join(lineas_limpias)

