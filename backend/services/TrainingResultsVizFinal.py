import matplotlib
matplotlib.use('Agg')  # Configurar backend no interactivo
import matplotlib.pyplot as plt
import os
import logging
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

logger = logging.getLogger(__name__)

class FinalModelVisualization:
    def __init__(self, name_model, y_true, y_pred, y_prob, accuracy_train, accuracy_test, precision_train, precision_test, recall_train, recall_test, f1_score_train, f1_score_test, feature_importances=None, feature_names=None):
        self.name_model = name_model
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.accuracy_train = accuracy_train
        self.accuracy_test = accuracy_test
        self.precision_train = precision_train
        self.precision_test = precision_test
        self.recall_train = recall_train
        self.recall_test = recall_test
        self.f1_score_train = f1_score_train
        self.f1_score_test = f1_score_test
        self.feature_importances = feature_importances
        self.feature_names = feature_names

    def generate_confusion_matrix(self, save_path):
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

    def generate_metrics_table(self, save_path):
        """Genera una tabla con las métricas del modelo final"""
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
                    f"{float(self.precision_train):.4f}",
                    f"{float(self.precision_test):.4f}"
                ],
                'Sensibilidad': [
                    f"{float(self.recall_train):.4f}",
                    f"{float(self.recall_test):.4f}"
                ],
                'F1-Score': [
                    f"{float(self.f1_score_train):.4f}",
                    f"{float(self.f1_score_test):.4f}"
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
            
            plt.title(f"Métricas del Modelo Final - {self.name_model}", pad=20)
            
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
            logger.error(f"Error generando tabla de métricas: {str(e)}")
            plt.close('all')

    def generate_roc_curve(self, save_path):
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

    def generate_feature_importance_plot(self, save_path):
        """Genera y guarda el gráfico de importancias de características"""
        try:
            plt.figure(figsize=(10, 6))
            # Asegúrate de que feature_importances y feature_names no sean None
            if self.feature_importances is not None and self.feature_names is not None:
                # Crear un DataFrame para las importancias
                importance_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': self.feature_importances
                })
                # Ordenar por importancia
                importance_df = importance_df.sort_values(by='Importance', ascending=False).head(15)

                # Crear el gráfico
                sns.barplot(x='Importance', y='Feature', data=importance_df, palette='Blues_d')
                plt.title(f'Top 15 Variables Más Importantes - {self.name_model}')
                plt.xlabel('Importancia')
                plt.ylabel('Características')
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
            else:
                logger.error("Las importancias de las características o los nombres son None.")
        except Exception as e:
            logger.error(f"Error generando gráfico de importancias: {str(e)}")
            plt.close() 

    @staticmethod
    def save_fine_tuning_results_to_csv(first_results, second_results, output_path):
        """Guarda los resultados del fine-tuning en un archivo CSV y retorna la ruta."""
        try:            
            # Crear DataFrame con los resultados
            results_df = pd.DataFrame({
                'Modelo': ['Primer Fine-Tuning', 'Segundo Fine-Tuning'],
                'Precisión': [first_results[0].precision, second_results[0].precision],
                'Recall': [first_results[0].recall, second_results[0].recall],
                'F1 Score': [first_results[0].f1_score, second_results[0].f1_score],
                'Exactitud': [first_results[0].accuracy, second_results[0].accuracy],
            })

            # Crear DataFrame para las diferencias
            differences_df = pd.DataFrame({
                'Métrica': ['Precisión', 'Recall', 'F1 Score', 'Exactitud'],
                'Diferencia': [
                    first_results[0].precision - second_results[0].precision,
                    first_results[0].recall - second_results[0].recall,
                    first_results[0].f1_score - second_results[0].f1_score,
                    first_results[0].accuracy - second_results[0].accuracy
                ]
            })

            # Verificar si el archivo CSV antiguo existe y cargarlo si es así
            old_results_df = None
            if os.path.exists(output_path):
                old_results_df = pd.read_csv(output_path)

            # Comparar con el CSV antiguo si existe
            if old_results_df is not None:
                if old_results_df.equals(results_df):
                    logger.info("No hay cambios en las métricas comparando con el CSV antiguo.")
                    return None

            # Guardar ambos DataFrames en un solo archivo CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                results_df.to_csv(f, index=False)
                f.write('\n\n\n')  # Agregar un salto de línea entre los dos DataFrames
                differences_df.to_csv(f, index=False)

            return output_path
        except Exception as e:
            logger.error(f"Error guardando resultados en CSV: {str(e)}")
            return None