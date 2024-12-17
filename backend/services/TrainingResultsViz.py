import matplotlib
matplotlib.use('Agg')  # Configurar backend no interactivo
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from typing import List
from backend.utils.response_models import TrainingResponseModel
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import seaborn as sns

logger = logging.getLogger(__name__)

class TrainingResultsVisualization:
    def __init__(self, name_model, accuracy_train, accuracy_test, precision=None, precision_train=None, recall=None, recall_train=None, f1_score=None, f1_score_train=None, y_true=None, y_pred=None, y_prob=None, feature_importances=None, feature_names=None):
        self.name_model = name_model
        self.accuracy_train = accuracy_train
        self.accuracy_test = accuracy_test
        self.precision = precision
        self.precision_train = precision_train
        self.recall = recall
        self.recall_train = recall_train
        self.f1_score = f1_score
        self.f1_score_train = f1_score_train
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.feature_importances = feature_importances
        self.feature_names = feature_names
        
    @staticmethod
    def generate_fine_tuning_results_table(training_results, output_dir, with_sentiment=True):
        """
        Genera una tabla con los resultados del fine-tuning y todas las visualizaciones
        """
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Primero generar la tabla comparativa de todos los modelos en la raíz
        comparison_data = {
            'Modelo': [],
            'Exactitud (Train)': [],
            'Exactitud (Test)': [],
            'Precisión (Train)': [],
            'Precisión (Test)': [],
            'Sensibilidad (Train)': [],
            'Sensibilidad (Test)': [],
            'F1-Score (Train)': [],
            'F1-Score (Test)': []
        }
        
        for result in training_results:
            comparison_data['Modelo'].append(result.name_model)
            comparison_data['Exactitud (Train)'].append(f"{float(result.accuracy_train):.4f}")
            comparison_data['Exactitud (Test)'].append(f"{float(result.accuracy):.4f}")
            comparison_data['Precisión (Train)'].append(f"{float(result.precision_train):.4f}")
            comparison_data['Precisión (Test)'].append(f"{float(result.precision):.4f}")
            comparison_data['Sensibilidad (Train)'].append(f"{float(result.recall_train):.4f}")
            comparison_data['Sensibilidad (Test)'].append(f"{float(result.recall):.4f}")
            comparison_data['F1-Score (Train)'].append(f"{float(result.f1_score_train):.4f}")
            comparison_data['F1-Score (Test)'].append(f"{float(result.f1_score):.4f}")
        
        # Crear DataFrame y tabla comparativa
        df_comparison = pd.DataFrame(comparison_data)
        plt.figure(figsize=(14, 6))
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('off')
        
        table = ax.table(
            cellText=df_comparison.values,
            colLabels=df_comparison.columns,
            cellLoc='center',
            loc='center',
            colColours=['#E49EDD'] * len(df_comparison.columns),
            cellColours=[['#F0F0F0' if i % 2 == 0 else '#FFFFFF'] * len(df_comparison.columns) 
                        for i in range(len(df_comparison))]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        table.auto_set_column_width(list(range(len(df_comparison.columns))))
        
        title = "Resultados de Fine-tuning (Con Análisis de Sentimiento)" if with_sentiment else "Resultados de Fine-tuning (Sin Análisis de Sentimiento)"
        plt.title(title, pad=20)
        
        # Guardar tabla comparativa en la raíz
        plt.savefig(
            os.path.join(output_dir, "fine_tuning_results.png"),
            bbox_inches='tight',
            dpi=300,
            pad_inches=0.5,
            facecolor='white'
        )
        plt.close()
        
        # Generar tabla comparativa con todas las métricas
        TrainingResultsVisualization._generate_all_metrics_comparison_table(training_results, output_dir)
        
        # Generar gráficos comparativos por métrica
        TrainingResultsVisualization._generate_metric_comparison_plots(training_results, output_dir)
        
        # Luego generar las visualizaciones individuales para cada modelo
        for result in training_results:
            try:
                # Crear directorio específico del modelo
                model_dir = os.path.join(output_dir, result.name_model.replace(" ", "_"))
                os.makedirs(model_dir, exist_ok=True)
                
                # Generar las visualizaciones
                viz = TrainingResultsVisualization(
                    name_model=result.name_model,
                    accuracy_train=result.accuracy_train,
                    accuracy_test=result.accuracy,
                    precision=result.precision,
                    precision_train=result.precision_train,
                    recall=result.recall,
                    recall_train=result.recall_train,
                    f1_score=result.f1_score,
                    f1_score_train=result.f1_score_train,
                    y_true=result.y_true,
                    y_pred=result.y_pred,
                    y_prob=result.y_prob,
                    feature_importances=result.feature_importances,
                    feature_names=result.feature_names
                )
                
                # Generar tabla detallada de métricas
                viz._generate_detailed_metrics_table(os.path.join(model_dir, "detailed_metrics_table.png"))
                
                # Generar visualizaciones en el directorio del modelo
                viz._generate_comparison_plot(os.path.join(model_dir, "comparison.png"))
                
                if result.y_true is not None and result.y_pred is not None:
                    viz._generate_confusion_matrix(os.path.join(model_dir, "confusion_matrix.png"))
                
                if result.y_true is not None and result.y_prob is not None:
                    try:
                        viz._generate_roc_curve(os.path.join(model_dir, "roc_curve.png"))
                    except Exception as e:
                        logger.error(f"Error generando curva ROC: {str(e)}")
                
                if result.feature_importances is not None and result.feature_names is not None:
                    viz._generate_feature_importance_plot(os.path.join(model_dir, "feature_importance.png"))
                    
            except Exception as e:
                logger.error(f"Error generando visualizaciones para {result.name_model}: {str(e)}")
                continue

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
                    'Precisión (Train)',
                    'Precisión (Test)',
                    'Sensibilidad (Train)',
                    'Sensibilidad (Test)',
                    'F1-Score (Train)',
                    'F1-Score (Test)'
                ],
                'Valor': [
                    f"{self.accuracy_train:.4f}",
                    f"{self.accuracy_test:.4f}",
                    f"{self.precision_train:.4f}" if self.precision_train else "N/A",
                    f"{self.precision:.4f}" if self.precision else "N/A",
                    f"{self.recall_train:.4f}" if self.recall_train else "N/A",
                    f"{self.recall:.4f}" if self.recall else "N/A",
                    f"{self.f1_score_train:.4f}" if self.f1_score_train else "N/A",
                    f"{self.f1_score:.4f}" if self.f1_score else "N/A"
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
            precision_values = [self.precision_train, self.precision]
            recall_values = [self.recall_train, self.recall]
            f1_values = [self.f1_score_train, self.f1_score]
            
            fig, ax = plt.subplots(2, 2, figsize=(12, 10))
            
            # Gráfico de Accuracy
            bars = ax[0, 0].bar(metrics, values, color=['#2ecc71', '#3498db'])
            ax[0, 0].set_title(f'Comparación de Accuracy - {self.name_model}')
            ax[0, 0].set_ylabel('Accuracy')
            for bar in bars:
                height = bar.get_height()
                ax[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.4f}',
                              ha='center', va='bottom')
            
            # Gráfico de Precisión
            bars = ax[0, 1].bar(metrics, precision_values, color=['#e74c3c', '#9b59b6'])
            ax[0, 1].set_title(f'Comparación de Precisión - {self.name_model}')
            ax[0, 1].set_ylabel('Precisión')
            for bar in bars:
                height = bar.get_height()
                ax[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.4f}',
                              ha='center', va='bottom')
            
            # Gráfico de Sensibilidad
            bars = ax[1, 0].bar(metrics, recall_values, color=['#f1c40f', '#e67e22'])
            ax[1, 0].set_title(f'Comparación de Sensibilidad - {self.name_model}')
            ax[1, 0].set_ylabel('Sensibilidad')
            for bar in bars:
                height = bar.get_height()
                ax[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.4f}',
                              ha='center', va='bottom')
            
            # Gráfico de F1-Score
            bars = ax[1, 1].bar(metrics, f1_values, color=['#1abc9c', '#34495e'])
            ax[1, 1].set_title(f'Comparación de F1-Score - {self.name_model}')
            ax[1, 1].set_ylabel('F1-Score')
            for bar in bars:
                height = bar.get_height()
                ax[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.4f}',
                              ha='center', va='bottom')
            
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

    def _generate_detailed_metrics_table(self, save_path):
        """
        Genera una tabla detallada con las métricas de entrenamiento y prueba
        """
        try:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
                
            # Crear DataFrame con las métricas detalladas
            data = {
                '': ['Entrenamiento', 'Prueba'],
                'Exactitud': [
                    f"{float(self.accuracy_train):.4f}",
                    f"{float(self.accuracy_test):.4f}"
                ],
                'Precisión': [
                    f"{float(self.precision_train):.4f}" if self.precision_train else "N/A",
                    f"{float(self.precision):.4f}" if self.precision else "N/A"
                ],
                'Sensibilidad': [
                    f"{float(self.recall_train):.4f}" if self.recall_train else "N/A",
                    f"{float(self.recall):.4f}" if self.recall else "N/A"
                ],
                'F1-Score': [
                    f"{float(self.f1_score_train):.4f}" if self.f1_score_train else "N/A",
                    f"{float(self.f1_score):.4f}" if self.f1_score else "N/A"
                ]
            }
            
            df = pd.DataFrame(data)
            
            # Crear figura y tabla
            plt.figure(figsize=(10, 4))
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.axis('off')
            
            # Crear tabla con el estilo definido
            table = ax.table(
                cellText=df.values,
                colLabels=df.columns,
                cellLoc='center',
                loc='center',
                colColours=['#E49EDD'] * len(df.columns),
                cellColours=[['#F0F0F0' if i % 2 == 0 else '#FFFFFF'] * len(df.columns) 
                            for i in range(len(df))]
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            table.auto_set_column_width(list(range(len(df.columns))))
            
            plt.title(f"Métricas Detalladas - {self.name_model}", pad=20)
            
            # Guardar figura
            plt.savefig(
                save_path,
                bbox_inches='tight',
                dpi=300,
                pad_inches=0.5,
                facecolor='white'
            )
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generando tabla de métricas detallada: {str(e)}")
            plt.close('all')

    @staticmethod
    def _generate_all_metrics_comparison_table(training_results, output_dir):
        """
        Genera una tabla comparativa con todas las métricas para todos los modelos
        """
        try:
            # Crear DataFrame con las métricas de prueba
            data = {
                'Modelo': [],
                'Exactitud': [],
                'Precisión': [],
                'Sensibilidad': [],
                'F1-score': [],
                'AUC': []
            }
            
            for result in training_results:
                data['Modelo'].append(result.name_model)
                data['Exactitud'].append(float(result.accuracy))
                data['Precisión'].append(float(result.precision))
                data['Sensibilidad'].append(float(result.recall))
                data['F1-score'].append(float(result.f1_score))
                # Calcular AUC si es posible
                if result.y_true is not None and result.y_prob is not None:
                    auc_score = roc_auc_score(result.y_true, result.y_prob)
                    data['AUC'].append(float(auc_score))
                else:
                    data['AUC'].append(np.nan)
            
            df = pd.DataFrame(data)
            
            # Crear figura y tabla con colores
            plt.figure(figsize=(12, 6))
            ax = sns.heatmap(df.drop(columns='Modelo').astype(float), annot=True, fmt=".2f", cmap='RdYlGn', cbar=False,
                             xticklabels=df.columns[1:], yticklabels=df['Modelo'], linewidths=0.5)
            
            plt.title("Comparación de Métricas entre Modelos", pad=35)
            plt.yticks(rotation=0)
            
            # Mover las etiquetas del eje X a la parte superior
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
            
            # Guardar tabla
            plt.savefig(
                os.path.join(output_dir, "all_metrics_comparison.png"),
                bbox_inches='tight',
                dpi=300,
                pad_inches=0.5,
                facecolor='white'
            )
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generando tabla comparativa de métricas: {str(e)}")
            plt.close('all')

    @staticmethod
    def _generate_metric_comparison_plots(training_results, output_dir):
        """
        Genera gráficos de barras comparativos para todas las métricas de prueba
        """
        try:
            # Crear directorio para gráficos
            graphics_dir = os.path.join(output_dir, "graficos")
            os.makedirs(graphics_dir, exist_ok=True)
            
            metrics = {
                'Exactitud': 'accuracy',
                'Precisión': 'precision',
                'Sensibilidad': 'recall',
                'F1-Score': 'f1_score'
            }
            
            # Función auxiliar para generar gráficos
            def create_metric_plot(results, save_dir, metric_name, metric_attr):
                plt.figure(figsize=(12, 6))
                
                models = [r.name_model for r in results]
                values = [float(getattr(r, metric_attr)) for r in results]
                
                # Cambiar el color de las barras
                bars = plt.bar(range(len(models)), values, color='#E49EDD')
                
                plt.title(f'{metric_name} de modelos')
                plt.xticks(range(len(models)), models, rotation=45, ha='right')
                plt.ylabel(metric_name)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Añadir valores sobre las barras
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.1%}',
                             ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(
                    os.path.join(save_dir, f"{metric_name.lower()}_comparison.png"),
                    bbox_inches='tight',
                    dpi=300,
                    facecolor='white'
                )
                plt.close()
            
            # Generar gráficos para cada métrica
            for metric_name, metric_attr in metrics.items():
                create_metric_plot(training_results, graphics_dir, metric_name, metric_attr)
                
        except Exception as e:
            logger.error(f"Error generando gráficos comparativos: {str(e)}")
            plt.close('all')