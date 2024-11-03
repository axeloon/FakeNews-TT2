import logging
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob

from backend.ETL.text_feature_extractor import TextProcessor
from backend.database.connection import engine

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
                "content_cant_oraciones", "content_cant_palabras", "content_cant_org", "content_cant_personas", "content_cant_loc"
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
                self._generate_boxplot(df, column, column_dir)

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

    def generate_frequency_graphics(self, csv_path: str):
        try:
            logger.info(f"Iniciando generación de gráficos de frecuencia para el CSV: {csv_path}")
            df = pd.read_csv(csv_path)

            frecuencia_columns = [
                "content_f_adjetivos", "content_f_verbos", "content_f_sustantivos",
                "content_f_pronombres", "content_f_adverbios", "content_f_determinantes",
                "content_f_stopwords", "content_f_puntuacion", "content_complejidad_lexica", "content_diversidad_lexica"
            ]

            base_dir = os.path.join(os.path.dirname(csv_path), "img")
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            for column in frecuencia_columns:
                column_dir = os.path.join(base_dir, column)
                if not os.path.exists(column_dir):
                    os.makedirs(column_dir)

                # 1. Histograma
                plt.figure(figsize=(10, 6))
                sns.histplot(df[column], bins=50, kde=True)
                plt.title(f'Histograma de {column}')
                plt.xlabel(column)
                plt.ylabel('Frecuencia')
                hist_path = os.path.join(column_dir, f"{column}_histograma.png")
                plt.savefig(hist_path)
                plt.close()
                logger.info(f"Histograma guardado en: {hist_path}")

                # 2. Gráfico de dispersión: frecuencia vs cantidad de palabras
                plt.figure(figsize=(10, 6))
                plt.scatter(df[column], df['content_cant_palabras'], alpha=0.5, color='blue')
                
                # Añadir línea de tendencia
                z = np.polyfit(df[column], df['content_cant_palabras'], 1)
                p = np.poly1d(z)
                plt.plot(df[column], p(df[column]), "r--", alpha=0.8)
                
                plt.xlabel(f'Frecuencia de {column}')
                plt.ylabel('Cantidad de palabras')
                plt.title(f'Relación entre {column} y cantidad de palabras')
                plt.grid(True, linestyle='--', alpha=0.3)
                
                scatter_path = os.path.join(column_dir, f"{column}_scatter_words.png")
                plt.savefig(scatter_path, bbox_inches='tight', dpi=300)
                plt.close()
                logger.info(f"Gráfico de dispersión guardado en: {scatter_path}")

                # 3. Boxplot
                self._generate_boxplot(df, column, column_dir)

                # 4. Gráfico de dispersión: frecuencia vs tipo de noticia
                plt.figure(figsize=(10, 6))

                # Crear el gráfico de dispersión con regresión
                sns.regplot(x=column, y='is_false', data=df,
                            scatter_kws={'alpha':0.5},
                            line_kws={'color': 'blue'})

                plt.xlabel(f'Frecuencia de {column}')
                plt.ylabel('Tipo')
                plt.title(f'Relación entre {column} y tipo de noticia')

                # Ajustar las etiquetas del eje y
                plt.yticks([0, 1], ['Verdadera', 'Falsa'])

                # Ajustar los límites del eje y
                plt.ylim(-0.2, 1.2)

                # Añadir grid para mejor lectura
                plt.grid(True, linestyle='--', alpha=0.3)

                scatter_type_path = os.path.join(column_dir, f"{column}_scatter_type.png")
                plt.savefig(scatter_type_path, bbox_inches='tight', dpi=300)
                plt.close()
                logger.info(f"Gráfico de dispersión por tipo guardado en: {scatter_type_path}")

            logger.info("Generación de gráficos de frecuencia completada exitosamente")
        except Exception as e:
            logger.error(f"Error en la generación de gráficos de frecuencia: {str(e)}")
            raise

    def filter_data(self, csv_path: str):
        try:
            logger.info(f"Cargando datos desde {csv_path}")
            df = pd.read_csv(csv_path)
            total_inicial = len(df)
            estadisticas_filtrado = {}

            # Diccionario de filtros y sus límites
            filtros = {
                'content_cant_oraciones': ('<=', 75),
                'content_cant_palabras': ('<=', 1800),
                'content_cant_org': ('<=', 5),
                'content_cant_personas': ('<=', 20),
                'content_cant_loc': ('<=', 10),
                'content_f_adjetivos': ('<=', 0.16),
                'content_f_verbos': ('<=', 0.25),
                'content_f_sustantivos': ('entre', 0.08, 0.35),
                'content_f_pronombres': ('<=', 0.13),
                'content_f_adverbios': ('<=', 0.10),
                'content_f_determinantes': ('entre', 0.03, 0.23),
                'content_f_stopwords': ('entre', 0.28, 0.62),
                'content_f_puntuacion': ('<=', 0.28)
            }

            # Aplicar cada filtro y registrar estadísticas
            for columna, condicion in filtros.items():
                registros_antes = len(df)
                
                if condicion[0] == '<=':
                    df_filtrado = df[df[columna] <= condicion[1]]
                    eliminados = registros_antes - len(df_filtrado)
                    estadisticas_filtrado[columna] = {
                        'eliminados': eliminados,
                        'porcentaje': round((eliminados/total_inicial) * 100, 2),
                        'condicion': f'mayor a {condicion[1]}'
                    }
                elif condicion[0] == 'entre':
                    df_filtrado = df[(df[columna] >= condicion[1]) & (df[columna] <= condicion[2])]
                    eliminados = registros_antes - len(df_filtrado)
                    estadisticas_filtrado[columna] = {
                        'eliminados': eliminados,
                        'porcentaje': round((eliminados/total_inicial) * 100, 2),
                        'condicion': f'fuera del rango [{condicion[1]}, {condicion[2]}]'
                    }
                
                df = df_filtrado

            # Eliminar la columna 'id' si existe
            if 'id' in df.columns:
                df = df.drop(columns=['id'])

            # Guardar el DataFrame en la base de datos
            df.to_sql(
                'noticias_redes_sociales_filtered',
                engine,
                if_exists='replace',
                index=False,
                chunksize=500
            )

            # Agregar estadísticas totales
            registros_finales = len(df)
            total_eliminados = total_inicial - registros_finales
            estadisticas_filtrado['resumen_total'] = {
                'registros_iniciales': total_inicial,
                'registros_finales': registros_finales,
                'total_eliminados': total_eliminados,
                'porcentaje_eliminado': round((total_eliminados/total_inicial) * 100, 2)
            }

            logger.info(f"Datos filtrados guardados exitosamente en la base de datos")
            return estadisticas_filtrado

        except Exception as e:
            logger.error(f"Error durante el filtrado de datos: {str(e)}")
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

    def _generate_boxplot(self, df, column, output_dir):
        """Genera un boxplot con estadísticas detalladas"""
        try:
            plt.figure(figsize=(15, 6))
            
            # Calcular estadísticas
            datos = df[column]
            q1 = datos.quantile(0.25)
            q2 = datos.median()
            q3 = datos.quantile(0.75)
            iqr = q3 - q1
            lower_bound = max(0, q1 - 1.5 * iqr)
            upper_bound = q3 + 1.5 * iqr
            
            # Identificar outliers
            outliers = datos[(datos < lower_bound) | (datos > upper_bound)]
            
            # Crear el boxplot
            bp = plt.boxplot(datos,
                            vert=False,
                            widths=0.7,
                            patch_artist=True,
                            boxprops=dict(facecolor='#a8e6cf', color='black'),
                            medianprops=dict(color='red', linewidth=2),
                            whiskerprops=dict(color='blue', linewidth=1.5),
                            capprops=dict(color='blue', linewidth=1.5),
                            flierprops=dict(marker='o', 
                                          markerfacecolor='gray',
                                          markersize=4,
                                          linestyle='none',
                                          markeredgecolor='gray'),
                            positions=[1],
                            showfliers=True,
                            whis=1.5)
            
            # Información sobre outliers y rango
            stats_text = ('Estadísticas:\n'
                         f'• Valores atípicos: {len(outliers)}\n'
                         f'• Rango normal: [{lower_bound:.2f}, {upper_bound:.2f}]\n'
                         f'• Q1: {q1:.2f}\n'
                         f'• Q2 (mediana): {q2:.2f}\n'
                         f'• Q3: {q3:.2f}')
            
            plt.text(1.02, 0.4, 
                    stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='center',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', 
                             facecolor='white', 
                             alpha=0.8,
                             pad=0.5))
            
            # Líneas verticales para los cuartiles
            plt.axvline(q1, color='green', linestyle='--', alpha=0.7)
            plt.axvline(q2, color='red', linestyle='--', alpha=0.7)
            plt.axvline(q3, color='purple', linestyle='--', alpha=0.7)
            
            plt.title(f'BoxPlot de {column}')
            plt.xlabel('Cantidad')
            plt.grid(True, linestyle='--', alpha=0.3)
            
            # Ajustar límites
            x_min = max(0, min(datos.min(), lower_bound) * 0.9)
            x_max = max(datos.max(), upper_bound) * 1.1
            plt.xlim(x_min, x_max)
            plt.ylim(0.5, 1.5)
            
            # Eliminar etiquetas del eje y
            plt.yticks([])
            
            # Leyenda
            handles = [
                plt.Rectangle((0,0),1,1, facecolor='#a8e6cf', label='Rango intercuartil'),
                plt.Line2D([], [], color='red', linestyle='--', label='Mediana (Q2)'),
                plt.Line2D([], [], color='green', linestyle='--', label='Primer cuartil (Q1)'),
                plt.Line2D([], [], color='purple', linestyle='--', label='Tercer cuartil (Q3)'),
                plt.Line2D([], [], color='blue', linestyle='-', label='Rango min-max'),
                plt.Line2D([], [], color='gray', marker='o', linestyle='none', label='Valores atípicos')
            ]
            plt.legend(handles=handles, 
                      loc='center left',
                      bbox_to_anchor=(1.02, 0.8),
                      title='Componentes del BoxPlot')
            
            plt.subplots_adjust(right=0.75)
            
            boxplot_path = os.path.join(output_dir, f"{column}_boxplot.png")
            plt.savefig(boxplot_path, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"BoxPlot guardado en: {boxplot_path}")
            
        except Exception as e:
            logger.error(f"Error generando boxplot: {str(e)}")
            raise


class SentimentAnalysis:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        # Verificar si la columna 'content' existe en el CSV
        if 'content' not in self.data.columns:
            raise ValueError("El archivo CSV debe contener una columna 'content'")
        # Calcular polaridad y subjetividad para cada texto en la columna 'content'
        self.data['polarity'] = self.data['content'].apply(self.calculate_polarity)
        self.data['subjectivity'] = self.data['content'].apply(self.calculate_subjectivity)
        # Categorizar polaridad y subjetividad
        self.data['polarity_category'] = self.data['polarity'].apply(self.categorize_polarity)
        self.data['subjectivity_category'] = self.data['subjectivity'].apply(self.categorize_subjectivity)

    def calculate_polarity(self, text: str) -> float:
        # Calcular polaridad usando TextBlob
        return TextBlob(text).sentiment.polarity

    def calculate_subjectivity(self, text: str) -> float:
        # Calcular subjetividad usando TextBlob
        return TextBlob(text).sentiment.subjectivity

    def categorize_polarity(self, polarity: float) -> int:
        if polarity >= 0.5:
            return 1  # Positivo
        elif polarity <= -0.5:
            return -1  # Negativo
        else:
            return 0  # Neutral

    def categorize_subjectivity(self, subjectivity: float) -> int:
        if subjectivity >= 0.6:
            return 1  # Subjetivo
        elif subjectivity <= 0.3:
            return -1  # Objetivo
        else:
            return 0  # Neutral

    def plot_histograms(self):
        base_dir = os.path.join(os.path.dirname(self.csv_path), "img")
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # Graficar histograma de polaridad
        plt.figure(figsize=(10, 6))
        plt.hist(self.data['polarity'], bins=20, edgecolor='black')
        plt.title('Histograma Polaridad')
        plt.xlabel('Polaridad')
        plt.ylabel('Frecuencia')
        hist_path_polarity = os.path.join(base_dir, 'histograma_polaridad.png')
        plt.savefig(hist_path_polarity)
        plt.close()
        logger.info(f"Histograma de polaridad guardado en: {hist_path_polarity}")

        # Graficar histograma de subjetividad
        plt.figure(figsize=(10, 6))
        plt.hist(self.data['subjectivity'], bins=20, edgecolor='black')
        plt.title('Histograma Objetividad')
        plt.xlabel('Objetividad')
        plt.ylabel('Frecuencia')
        hist_path_subjectivity = os.path.join(base_dir, 'histograma_objetividad.png')
        plt.savefig(hist_path_subjectivity)
        plt.close()
        logger.info(f"Histograma de objetividad guardado en: {hist_path_subjectivity}")

    def save_csv(self, output_path: str):
        self.data.to_csv(output_path, index=False)
        logger.info(f"CSV guardado en: {output_path}")