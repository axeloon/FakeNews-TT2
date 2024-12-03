import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class TrainingResultsVisualization:
    def __init__(self, name_model, accuracy_train, accuracy_test, feature_importances=None, feature_names=None):
        self.name_model = name_model
        self.accuracy_train = accuracy_train
        self.accuracy_test = accuracy_test
        self.feature_importances = feature_importances
        self.feature_names = feature_names

    def generate_file_paths(self, base_path):
        """Genera rutas de archivo para las visualizaciones y crea las subcarpetas necesarias"""
        metrics_filename = os.path.join(base_path, f"{self.name_model.replace(' ', '_')}_metrics")
        importance_filename = os.path.join(base_path, f"{self.name_model.replace(' ', '_')}_feature_importance")
        comparison_filename = os.path.join(base_path, f"{self.name_model.replace(' ', '_')}_comparison")
        
        # Crear las subcarpetas necesarias si no están creadas
        os.makedirs(base_path, exist_ok=True)
        
        return metrics_filename, importance_filename, comparison_filename

    def generate_accuracy_tables(self, metrics_filename):
        """Genera una tabla similar a la mostrada en la imagen"""
        # Crear DataFrame con el formato deseado
        data = {
            'Modelo': [self.name_model],
            'Entrenamiento': [f"{self.accuracy_train:.2f}"],
            'Prueba': [f"{self.accuracy_test:.2f}"]
        }
        df = pd.DataFrame(data)

        # Configurar el estilo de la tabla
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.axis('off')
        
        # Crear tabla con estilo mejorado
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f2f2f2'] * len(df.columns),
            cellColours=[['#ffffff'] * len(df.columns)]
        )
        
        # Ajustar el estilo de la tabla
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Ajustar bordes y colores
        for cell in table._cells:
            table._cells[cell].set_edgecolor('#cccccc')
            table._cells[cell].set_linewidth(1)
        
        # Guardar la tabla
        plt.savefig(f"{metrics_filename}.png", 
                    bbox_inches='tight', 
                    dpi=300,
                    facecolor='white',
                    edgecolor='none',
                    pad_inches=0.1)
        plt.close()
        
        return fig

    def generate_feature_importance_plot(self):
        """Genera un gráfico de barras con la importancia de las variables"""
        plt.figure(figsize=(10, 6))
        
        if self.feature_importances is None or self.feature_names is None:
            # Mensaje informativo si no hay importancias
            plt.text(0.5, 0.5,
                     f'Modelo: {self.name_model}\n\n'
                     'No se pudieron calcular las importancias de características.\n'
                     'Se recomienda usar técnicas como Permutation Importance.',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=plt.gca().transAxes,
                     fontsize=12,
                     bbox=dict(facecolor='#e49edd', alpha=0.1))
            plt.axis('off')
        else:
            # Tomar solo las top 15 variables más importantes
            indices = np.argsort(self.feature_importances)[-15:][::-1]
            sorted_importances = np.array(self.feature_importances)[indices]
            sorted_features = np.array(self.feature_names)[indices]
            
            plt.barh(sorted_features, sorted_importances, align='center')
            plt.xlabel('Importancia')
            plt.title(f'Top 15 variables más importantes - {self.name_model}')
            plt.gca().invert_yaxis()
        
        plt.tight_layout()
        return plt.gcf()

    def save_accuracy_tables_to_csv(self, base_path):
        """Guarda la tabla de métricas en un archivo CSV"""
        metrics_filename, _, _ = self.generate_file_paths(base_path)
        self.generate_accuracy_tables(metrics_filename)

    def save_feature_importance_plot(self, base_path):
        """Guarda el gráfico de importancia de variables en un archivo"""
        _, importance_filename, _ = self.generate_file_paths(base_path)
        if self.feature_importances is not None and self.feature_names is not None:
            fig = self.generate_feature_importance_plot()
            fig.savefig(f"{importance_filename}.png", bbox_inches='tight', dpi=300)
            plt.close(fig)

    def generate_and_save_all(self, base_path):
        """Genera y guarda todas las visualizaciones"""
        try:
            logger.info(f"Generando visualizaciones en {base_path}")
            
            # Crear directorios si no existen
            os.makedirs(base_path, exist_ok=True)
            
            # Generar y guardar tabla de métricas
            metrics_filename, importance_filename, comparison_filename = self.generate_file_paths(base_path)
            logger.info("Generando tabla de métricas...")
            self.generate_accuracy_tables(metrics_filename)
            
            # Generar y guardar gráfico de importancia de características
            if self.feature_importances is not None and self.feature_names is not None:
                logger.info("Generando gráfico de importancia de características...")
                self.save_feature_importance_plot(base_path)
            else:
                logger.warning("No hay datos de importancia de características disponibles")
            
            # Generar y guardar gráfico comparativo
            logger.info("Generando gráfico comparativo...")
            self.save_comparison_plot(base_path)
            
            logger.info(f"Todas las visualizaciones generadas exitosamente en {base_path}")
        except Exception as e:
            logger.error(f"Error al generar visualizaciones: {str(e)}")
            raise

    @staticmethod
    def generate_fine_tuning_results_table(fine_tuning_results, base_path):
        """Genera una tabla comparativa con todos los resultados del fine-tuning"""
        plt.figure(figsize=(8, 4))  # Reducido el tamaño vertical
        
        # Extraer datos de los resultados
        model_names = []
        train_accuracies = []
        test_accuracies = []
        
        for result in fine_tuning_results:
            model_names.append(result.name_model)
            train_accuracies.append(result.accuracy_train)
            test_accuracies.append(result.accuracy)
        
        # Crear tabla
        cell_text = [
            [f"{train:.2f}", f"{test:.2f}"]
            for train, test in zip(train_accuracies, test_accuracies)
        ]
        
        table = plt.table(
            cellText=cell_text,
            rowLabels=model_names,
            colLabels=['Entrenamiento', 'Prueba'],
            loc='center',
            cellLoc='center'
        )
        
        # Personalizar apariencia
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.2)  # Reducida la escala vertical
        
        # Configurar colores
        for i in range(len(model_names)):
            table[(i+1, 0)].set_facecolor('#e49edd')
            table[(i+1, 1)].set_facecolor('#e49edd')
        
        # Configurar título
        plt.title('Exactitud en Fine-tuning', pad=10)  # Reducido el padding
        plt.axis('off')
        
        # Guardar figura con menos espacio en blanco
        plt.savefig(
            f"{base_path}/fine_tuning_results_table.png",
            bbox_inches='tight',
            dpi=300,
            facecolor='white',
            edgecolor='none',
            pad_inches=0.1  # Reducido el padding exterior
        )
        plt.close()

    def generate_comparison_plot(self):
        """Genera un gráfico comparativo de exactitud entre entrenamiento y prueba"""
        plt.figure(figsize=(8, 6))
        metrics = ['Exactitud']
        train_values = [self.accuracy_train]
        test_values = [self.accuracy_test]

        x = np.arange(len(metrics))
        width = 0.35

        plt.bar(x - width/2, train_values, width, label='Entrenamiento')
        plt.bar(x + width/2, test_values, width, label='Prueba')

        plt.ylabel('Valor')
        plt.title(f'Comparación de métricas - {self.name_model}')
        plt.xticks(x, metrics)
        plt.legend()
        plt.tight_layout()
        
        return plt.gcf()

    def save_comparison_plot(self, base_path):
        """Guarda el gráfico comparativo en un archivo"""
        _, _, comparison_filename = self.generate_file_paths(base_path)
        fig = self.generate_comparison_plot()
        fig.savefig(f"{comparison_filename}.png", bbox_inches='tight', dpi=300)
        plt.close(fig)

    def generate_fine_tuning_comparison(self, original_accuracy_train, original_accuracy_test,
                                      fine_tuned_accuracy_train, fine_tuned_accuracy_test,
                                      save_path):
        """Genera un gráfico comparativo entre el modelo original y el fine-tuned"""
        plt.figure(figsize=(10, 6))
        
        # Configurar las barras
        labels = ['Entrenamiento', 'Prueba']
        original_values = [original_accuracy_train, original_accuracy_test]
        fine_tuned_values = [fine_tuned_accuracy_train, fine_tuned_accuracy_test]
        
        x = np.arange(len(labels))
        width = 0.35
        
        # Crear las barras
        plt.bar(x - width/2, original_values, width, label='Modelo Original', color='#e49edd')
        plt.bar(x + width/2, fine_tuned_values, width, label='Modelo Ajustado', color='#9ee4dd')
        
        # Personalizar el gráfico
        plt.ylabel('Exactitud')
        plt.title(f'Comparación de Exactitud - {self.name_model}')
        plt.xticks(x, labels)
        plt.legend()
        
        # Agregar valores sobre las barras
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
        
        autolabel(plt.gca().patches[:2])
        autolabel(plt.gca().patches[2:])
        
        # Ajustar el diseño y guardar
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()