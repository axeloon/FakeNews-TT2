import spacy
import logging
import pandas as pd
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self):
        logger.info("Inicializando TextProcessor...")
        try:
            self.nlp = spacy.load("es_core_news_sm")
            self.data = None
            self.features = None
        except Exception as e:
            logger.error(f"Error cargando modelo spaCy: {str(e)}")
            raise

    def load_data(self, csv_path: str):
        """Carga el dataset desde un archivo CSV"""
        try:
            self.data = pd.read_csv(csv_path)
            self.data['content'] = self.data['content'].str.lower()
            logger.info(f"Datos cargados exitosamente desde {csv_path}")
        except Exception as e:
            logger.error(f"Error cargando el archivo CSV: {str(e)}")
            raise

    def extract_features(self) -> pd.DataFrame:
        """Extrae características tanto del título como del contenido"""
        if self.data is None:
            raise ValueError("El dataset no se ha cargado. Ejecute 'load_data()' primero.")

        logger.info("Iniciando extracción de características del dataset...")
        
        # Procesar contenido
        content_features = self.data['content'].apply(self._extract_features_from_text)
        content_features_df = pd.json_normalize(content_features).add_prefix('content_')
        logger.info(f"Características del contenido extraídas exitosamente: {content_features_df.head()}")
        # Combinar características con datos originales
        result_df = pd.concat([self.data, content_features_df], axis=1)
        logger.info("Extracción de características completada.")
        return result_df

    def _extract_features_from_text(self, text: str) -> Dict[str, Any]:
        """Extrae características de un texto individual"""
        try:
            doc = self.nlp(str(text))  # Asegurar que el texto sea string
            
            features = {
                "cant_oraciones": len(list(doc.sents)),
                "cant_palabras": len(doc),
                "cant_org": len([ent for ent in doc.ents if ent.label_ == "ORG"]),
                "cant_personas": len([ent for ent in doc.ents if ent.label_ == "PER"]),
                "cant_fechas": len([ent for ent in doc.ents if ent.label_ == "DATE"]),
                "cant_loc": len([ent for ent in doc.ents if ent.label_ == "LOC"]),
                "cant_eventos": len([ent for ent in doc.ents if ent.label_ == "EVENT"]),
                "f_adjetivos": len([token for token in doc if token.pos_ == "ADJ"]) / len(doc) if len(doc) > 0 else 0,
                "f_verbos": len([token for token in doc if token.pos_ == "VERB"]) / len(doc) if len(doc) > 0 else 0,
                "f_sustantivos": len([token for token in doc if token.pos_ == "NOUN"]) / len(doc) if len(doc) > 0 else 0,
                "f_pronombres": len([token for token in doc if token.pos_ == "PRON"]) / len(doc) if len(doc) > 0 else 0,
                "f_adverbios": len([token for token in doc if token.pos_ == "ADV"]) / len(doc) if len(doc) > 0 else 0,
                "f_determinantes": len([token for token in doc if token.pos_ == "DET"]) / len(doc) if len(doc) > 0 else 0,
                "f_stopwords": len([token for token in doc if token.is_stop]) / len(doc) if len(doc) > 0 else 0,
                "f_puntuacion": len([token for token in doc if token.is_punct]) / len(doc) if len(doc) > 0 else 0,
                "complejidad_lexica": len(set([token.text.lower() for token in doc if not token.is_punct])) / len(doc) if len(doc) > 0 else 0,
                "longitud_promedio_palabras": sum(len(token.text) for token in doc if not token.is_punct) / len([token for token in doc if not token.is_punct]) if len(doc) > 0 else 0,
                "diversidad_lexica": len(set([token.text.lower() for token in doc if not token.is_punct])) / len(doc) if len(doc) > 0 else 0
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error procesando texto: {str(e)}")
            raise

    def get_stopwords(self):
        """Devuelve las stopwords utilizadas por el modelo spaCy en formato JSON ordenado"""
        stopwords = sorted(list(self.nlp.Defaults.stop_words))
        return {
            "stopwords": stopwords,
            "total": len(stopwords),
            "metadata": {
                "idioma": "es",
                "modelo": "spacy es_core_news_sm"
            }
        }
