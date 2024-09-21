import os
import shutil
import logging

logger = logging.getLogger(__name__)

class FileService:
    @staticmethod
    def clean_folders(folders: list[str]) -> None:
        """
        Elimina las carpetas especificadas y su contenido.

        Args:
            folders (list[str]): Lista de rutas de carpetas a eliminar.
        """
        for folder in folders:
            try:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                    logger.info(f"Carpeta eliminada: {folder}")
                else:
                    logger.warning(f"La carpeta no existe: {folder}")
            except Exception as e:
                logger.error(f"Error al eliminar la carpeta {folder}: {str(e)}")

    @staticmethod
    def upload_folders(folders: list[str]) -> None:
        """
        Esquema básico para subir carpetas.

        Args:
            folders (list[str]): Lista de rutas de carpetas a subir.
        """
        for folder in folders:
            try:
                # Aquí iría la lógica para subir la carpeta
                # Por ejemplo, usando una API de almacenamiento en la nube
                logger.info(f"Simulando subida de carpeta: {folder}")
            except Exception as e:
                logger.error(f"Error al subir la carpeta {folder}: {str(e)}")

