import logging
from backend.services.EmolService import EmolService

logger = logging.getLogger(__name__)

emol_service = EmolService()

async def scrape_emol_historic(id_fin: int, cantidad_noticias: int):
    noticias_filtradas = []
    id_inicio = id_fin - cantidad_noticias

    for id_noticia in range(id_inicio, id_fin + 1):
        logger.info(f"Scrappeando noticia ID: {id_noticia}")
        noticia = await emol_service.obtener_noticia(id_noticia)
        if noticia and '_source' in noticia:
            source = noticia['_source']
            if emol_service.es_noticia_de_interes(source):
                noticia_procesada = emol_service.procesar_noticia(source)
                if noticia_procesada:
                    noticias_filtradas.append(noticia_procesada)

    logger.info(f"Proceso completado. Total de noticias filtradas: {len(noticias_filtradas)}")
    return noticias_filtradas