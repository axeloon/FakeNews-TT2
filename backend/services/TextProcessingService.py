import logging
from backend.ETL.text_feature_extractor import TextProcessor

# Configurar el logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessingService:
    def __init__(self):
        self.processor = TextProcessor()

    def process_csv(self, csv_path: str):
        try:
            logger.info(f"Iniciando procesamiento del CSV: {csv_path}")
            
            # Cargar y procesar el CSV
            self.processor.load_data(csv_path)
            features_df = self.processor.extract_features()
            
            logger.info("Procesamiento del CSV completado exitosamente")
            # Guardar el DataFrame en un archivo CSV
            output_path = "backend/ETL/processed_features.csv"
            features_df.to_csv(output_path, index=False)
            logger.info(f"DataFrame guardado exitosamente en: {output_path}")
            # Reemplazar valores NaN con 0 antes de retornar
            features_df = features_df.fillna(0)
            logger.info("Valores NaN reemplazados con 0")
            return features_df
        except Exception as e:
            logger.error(f"Error en el procesamiento del CSV: {str(e)}")
            raise

    def dump_stopwords(self):
        try:
            logger.info("Extrayendo stopwords del modelo spaCy...")
            self.processor = TextProcessor()
            stopwords = self.processor.get_stopwords()
            logger.info("Stopwords extraídas exitosamente.")
            return stopwords
        except Exception as e:
            logger.error(f"Error al extraer stopwords: {str(e)}")
            raise