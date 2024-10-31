import logging
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
            output_path = csv_path.replace("_raw.csv", "_processed_features.csv")
            features_df.to_csv(output_path, index=False)
            logger.info(f"DataFrame guardado exitosamente en: {output_path}")
            # Reemplazar valores NaN con 0 antes de retornar
            features_df = features_df.fillna(0)
            logger.info("Valores NaN reemplazados con 0")
            return features_df
        except Exception as e:
            logger.error(f"Error en el procesamiento del CSV: {str(e)}")
            raise

    def calculate_statistics(self, csv_path: str):
        try:
            logger.info(f"Iniciando cálculo de estadísticas para el CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            stats = {}

            for column in df.columns:
                if df[column].dtype in ['int64', 'float64']:
                    stats[column] = {
                        "valor_minimo": df[column].min(),
                        "valor_maximo": df[column].max(),
                        "mediana": df[column].median(),
                        "datos_iguales_a_cero": (df[column] == 0).mean() * 100,
                        "valores_distintos": df[column].nunique()
                    }

            # Convertir valores de numpy a tipos nativos de Python
            for key, value in stats.items():
                stats[key] = {k: (int(v) if isinstance(v, (np.int64, np.int32)) else float(v) if isinstance(v, (np.float64, np.float32)) else v) for k, v in value.items()}

            logger.info("Cálculo de estadísticas completado exitosamente")
            return stats
        except Exception as e:
            logger.error(f"Error en el cálculo de estadísticas: {str(e)}")
            raise

    def calculate_correlation(self, csv_path: str):
        try:
            logger.info(f"Iniciando cálculo de correlación para el CSV: {csv_path}")
            df = pd.read_csv(csv_path)

            selected_columns = [
                "content_cant_oraciones", "content_cant_palabras", "content_cant_org", "content_cant_personas", "content_cant_loc",
                "content_f_adjetivos", "content_f_verbos", "content_f_sustantivos", "content_f_pronombres", "content_f_adverbios", "content_f_determinantes",
                "content_f_stopwords", "content_f_puntuacion", "content_complejidad_lexica", "content_diversidad_lexica"
            ]

            df = df[selected_columns]

            # Calcular la matriz de correlación
            correlacion = df.corr()

            # Filtrar solo las correlaciones que sean significativas (ejemplo: mayores a 0.2 o menores a -0.2)
            correlacion_filtrada = correlacion[(correlacion > 0.2) | (correlacion < -0.2)]

            # Guardar la imagen del heatmap simplificado
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlacion_filtrada, annot=True, cmap='coolwarm', center=0, linewidths=0.5, fmt=".2f", cbar_kws={'label': 'Correlación'})
            plt.title('Matriz de Correlación Simplificada')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            output_image_path = csv_path.replace(".csv", "_correlation_heatmap_simplified.png")
            plt.savefig(output_image_path)
            logger.info(f"Imagen de la correlación guardada exitosamente en: {output_image_path}")
            
            return output_image_path
        except Exception as e:
            logger.error(f"Error en el cálculo de correlación: {str(e)}")
            raise

    def generate_graphics(self, csv_path: str):
        try:
            logger.info(f"Iniciando generación de gráficos para el CSV: {csv_path}")
            df = pd.read_csv(csv_path)

            selected_columns = [
                "content_cant_oraciones", "content_cant_palabras", "content_cant_org", "content_cant_personas", "content_cant_loc",
                "content_f_adjetivos", "content_f_verbos", "content_f_sustantivos", "content_f_pronombres", "content_f_adverbios", "content_f_determinantes",
                "content_f_stopwords", "content_f_puntuacion", "content_complejidad_lexica", "content_diversidad_lexica"
            ]

            base_dir = os.path.join(os.path.dirname(csv_path), "img")
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            for column in selected_columns:
                column_dir = os.path.join(base_dir, column)
                if not os.path.exists(column_dir):
                    os.makedirs(column_dir)

                # Histograma
                plt.figure(figsize=(10, 6))
                sns.histplot(df[column], bins=50, kde=True)
                plt.title(f'Histograma de {column}')
                plt.xlabel(column)
                plt.ylabel('Frecuencia')
                hist_path = os.path.join(column_dir, f"{column}_histograma.png")
                plt.savefig(hist_path)
                plt.close()
                logger.info(f"Histograma guardado en: {hist_path}")

                # BoxPlot
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=df[column])
                plt.title(f'BoxPlot de {column}')
                boxplot_path = os.path.join(column_dir, f"{column}_boxplot.png")
                plt.savefig(boxplot_path)
                plt.close()
                logger.info(f"BoxPlot guardado en: {boxplot_path}")

                # Valores más comunes
                valores_comunes = df[column].value_counts().reset_index().head(10)
                valores_comunes.columns = [column, 'Cantidad']
                valores_comunes['Frecuencia'] = ((valores_comunes['Cantidad'] / valores_comunes['Cantidad'].sum()) * 100).round(2).astype(str) + "%"
                valores_comunes[column] = valores_comunes[column].astype(int)
                valores_comunes['Cantidad'] = valores_comunes['Cantidad'].astype(int)
                valores_comunes_path = os.path.join(column_dir, f"{column}_valores_comunes.png")

                # Crear la tabla de valores más comunes como imagen con formato y color
                fig, ax = plt.subplots(figsize=(6, len(valores_comunes) * 0.6))
                ax.axis('off')
                table = ax.table(
                    cellText=valores_comunes.values,
                    colLabels=valores_comunes.columns,
                    cellLoc='center',
                    loc='center',
                    colColours=['#e49edd', '#e49edd', '#e49edd']
                )
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.auto_set_column_width(col=list(range(len(valores_comunes.columns))))
                plt.savefig(valores_comunes_path, bbox_inches='tight')
                plt.close()
                logger.info(f"Valores más comunes guardado en: {valores_comunes_path}")

            logger.info("Generación de gráficos completada exitosamente")
        except Exception as e:
            logger.error(f"Error en la generación de gráficos: {str(e)}")
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