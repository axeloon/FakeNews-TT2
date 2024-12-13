import matplotlib
matplotlib.use('Agg')  # Configurar backend no interactivo
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from typing import List
from backend.utils.response_models import TrainingResponseModel
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

logger = logging.getLogger(__name__)

class TrainingResultsVisualization:
    def __init__(self, name_model, accuracy_train, accuracy_test, precision=None, recall=None, f1_score=None, y_true=None, y_pred=None, y_prob=None, feature_importances=None, feature_names=None):
        self.name_model = name_model
        self.accuracy_train = accuracy_train
        self.accuracy_test = accuracy_test
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.feature_importances = feature_importances
        self.feature_names = feature_names
        
    def generate_and_save_all(self, base_path):
        """Genera y guarda todas las visualizaciones"""
        try:
            logger.info(f"Generando visualizaciones en {base_path}")
            os.makedirs(base_path, exist_ok=True)
            
            # Generar tabla de métricas
            logger.info("Generando tabla de métricas...")
            self._generate_metrics_table(os.path.join(base_path, "metrics_table.png"))
            
            # Generar gráfico de importancia de características si está disponible
            if self.feature_importances is not None and self.feature_names is not None:
                logger.info("Generando gráfico de importancia de características...")
                self._generate_feature_importance_plot(os.path.join(base_path, "feature_importance.png"))
            else:
                logger.warning("No hay datos de importancia de características disponibles")
            
            # Generar gráfico comparativo
            logger.info("Generando gráfico comparativo...")
            self._generate_comparison_plot(os.path.join(base_path, "comparison.png"))
            
            # Nuevas visualizaciones
            if self.y_true is not None and self.y_pred is not None:
                self._generate_confusion_matrix(os.path.join(base_path, "confusion_matrix.png"))
            
            if self.y_true is not None and self.y_prob is not None:
                self._generate_roc_curve(os.path.join(base_path, "roc_curve.png"))
            
            logger.info(f"Todas las visualizaciones generadas exitosamente en {base_path}")
            plt.close('all')  # Cerrar todas las figuras
            
        except Exception as e:
            logger.error(f"Error generando visualizaciones: {str(e)}")
            plt.close('all')  # Asegurarse de cerrar las figuras incluso si hay error
            raise

    @staticmethod
    def generate_fine_tuning_results_table(results: List[TrainingResponseModel], save_path: str):
        try:
            # Verificar que hay resultados para procesar
            if not results:
                logger.warning("No hay resultados para generar la tabla")
                return
                
            plt.style.use('default')
            
            # Crear DataFrame con los resultados
            data = {
                'Modelo': [],
                'Entrenamiento': [],
                'Prueba': [],
                'F1-Score': []  # Añadimos F1-Score para más información
            }
            
            for result in results:
                # Limpiar el nombre del modelo para mejor visualización
                model_name = result.name_model.replace(" (Fine-Tuned)", "").replace("_", " ")
                data['Modelo'].append(model_name)
                data['Entrenamiento'].append(f"{result.accuracy_train:.4f}")
                data['Prueba'].append(f"{result.accuracy:.4f}")
                data['F1-Score'].append(f"{result.f1_score:.4f}")
            
            df = pd.DataFrame(data)
            
            # Crear figura y tabla
            fig, ax = plt.subplots(figsize=(12, len(df) + 2))
            ax.axis('tight')
            ax.axis('off')
            
            # Crear tabla con colores alternados
            table = ax.table(cellText=df.values,
                           colLabels=df.columns,
                           cellLoc='center',
                           loc='center',
                           colColours=['#e6e6e6']*len(df.columns),
                           cellColours=[['#f2f2f2' if i%2==0 else 'white']*len(df.columns) 
                                      for i in range(len(df))])
            
            # Ajustar formato
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # Determinar el título basado en el path
            title = "Resultados de Fine-tuning"
            if "no_sentiment" in save_path:
                title += " (Sin Análisis de Sentimiento)"
            else:
                title += " (Con Análisis de Sentimiento)"
                
            plt.title(title, pad=20)
            
            # Asegurar que el directorio existe
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Guardar figura
            plt.savefig(os.path.join(save_path, 'fine_tuning_results.png'),
                       bbox_inches='tight',
                       dpi=300,
                       facecolor='white')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generando tabla de resultados: {str(e)}")
            plt.close('all')
            # No levantar la excepción para permitir que el proceso continúe

    def _generate_metrics_table(self, save_path):
        try:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
                
            plt.style.use('default')
            
            # Crear DataFrame con todas las métricas
            data = {
                'Métrica': [
                    'Exactitud (Train)', 
                    'Exactitud (Test)',
                    'Precisión',
                    'Sensibilidad',
                    'F1-Score'
                ],
                'Valor': [
                    f"{self.accuracy_train:.4f}",
                    f"{self.accuracy_test:.4f}",
                    f"{self.precision:.4f}" if hasattr(self, 'precision') else "N/A",
                    f"{self.recall:.4f}" if hasattr(self, 'recall') else "N/A",
                    f"{self.f1_score:.4f}" if hasattr(self, 'f1_score') else "N/A"
                ]
            }
            df = pd.DataFrame(data)
            
            # Crear figura y tabla
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.axis('tight')
            ax.axis('off')
            
            # Crear tabla con colores alternados
            table = ax.table(cellText=df.values,
                            colLabels=df.columns,
                            cellLoc='center',
                            loc='center',
                            colColours=['#e6e6e6']*2,
                            cellColours=[['#f2f2f2', 'white'] for _ in range(len(df))])
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            plt.title(f"Métricas para {self.name_model}", pad=20)
            
            # Guardar figura
            plt.savefig(save_path,
                       bbox_inches='tight',
                       dpi=300,
                       facecolor='white')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generando tabla de métricas: {str(e)}")
            plt.close('all')

    def _generate_feature_importance_plot(self, save_path):
        try:
            plt.style.use('default')
            
            if self.feature_importances is not None and self.feature_names is not None:
                # Convertir a numpy arrays si no lo son
                importances = np.array(self.feature_importances)
                names = np.array(self.feature_names)
                
                # Ordenar por importancia
                sorted_idx = np.argsort(importances)
                pos = np.arange(len(sorted_idx)) + .5
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(pos, importances[sorted_idx])
                ax.set_yticks(pos)
                ax.set_yticklabels(names[sorted_idx])
                ax.set_xlabel('Importancia Relativa')
                ax.set_title(f'Importancia de Características - {self.name_model}')
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
        except Exception as e:
            logger.error(f"Error generando gráfico de importancia: {str(e)}")
            plt.close('all')
            raise

    def _generate_comparison_plot(self, save_path):
        try:
            plt.style.use('default')
            
            metrics = ['Train', 'Test']
            values = [self.accuracy_train, self.accuracy_test]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(metrics, values, color=['#2ecc71', '#3498db'])
            
            # Añadir valores sobre las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom')
            
            ax.set_ylim(0, 1.0)
            ax.set_title(f'Comparación de Accuracy - {self.name_model}')
            ax.set_ylabel('Accuracy')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generando gráfico comparativo: {str(e)}")
            plt.close('all')
            raise

    def _generate_confusion_matrix(self, save_path):
        """Genera y guarda la matriz de confusión"""
        try:
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(self.y_true, self.y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Verdadero', 'Falso'],
                       yticklabels=['Verdadero', 'Falso'])
            
            plt.title(f'Matriz de Confusión - {self.name_model}')
            plt.ylabel('Valor Real')
            plt.xlabel('Valor Predicho')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generando matriz de confusión: {str(e)}")
            plt.close()

    def _generate_roc_curve(self, save_path):
        """Genera y guarda la curva ROC"""
        try:
            plt.figure(figsize=(8, 6))
            
            fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title(f'Curva ROC - {self.name_model}')
            plt.legend(loc="lower right")
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generando curva ROC: {str(e)}")
            plt.close()