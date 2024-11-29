import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class TrainingResultsVisualization:
    def __init__(self, name_model, accuracy_train, accuracy_test, feature_importances=None, feature_names=None):
        self.name_model = name_model
        self.accuracy_train = accuracy_train
        self.accuracy_test = accuracy_test
        self.feature_importances = feature_importances
        self.feature_names = feature_names

    def generate_file_paths(self, base_path):
        """Genera rutas de archivo para las visualizaciones y crea las subcarpetas necesarias si no están creadas"""
        train_filename = os.path.join(base_path, f"{self.name_model.replace(' ', '_')}_accuracy_train")
        test_filename = os.path.join(base_path, f"{self.name_model.replace(' ', '_')}_accuracy_test")
        importance_filename = os.path.join(base_path, f"{self.name_model.replace(' ', '_')}_feature_importance")
        comparison_filename = os.path.join(base_path, f"{self.name_model.replace(' ', '_')}_comparison")
        
        # Crear las subcarpetas necesarias si no están creadas
        os.makedirs(base_path, exist_ok=True)
        
        return train_filename, test_filename, importance_filename, comparison_filename

    def generate_accuracy_tables(self, train_filename, test_filename):
        """Genera dos tablas comparativas como imágenes: una para entrenamiento y otra para prueba"""
        # Tabla de métricas de entrenamiento
        train_data = {
            'Modelo': [self.name_model],
            'Exactitud en Entrenamiento': [round(self.accuracy_train, 2)],
        }
        train_df = pd.DataFrame(train_data)

        # Tabla de métricas de prueba 
        test_data = {
            'Modelo': [self.name_model],
            'Exactitud en Prueba': [round(self.accuracy_test, 2)],
        }
        test_df = pd.DataFrame(test_data)

        # Generar imagen para tabla de entrenamiento
        fig_train, ax_train = plt.subplots(figsize=(6, len(train_df) * 0.6))
        ax_train.axis('off')
        table_train = ax_train.table(
            cellText=train_df.values,
            colLabels=train_df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#e49edd', '#e49edd']
        )
        table_train.auto_set_font_size(False)
        table_train.set_fontsize(10)
        table_train.auto_set_column_width(col=list(range(len(train_df.columns))))
        output_image_path = f"{train_filename}.png"
        plt.savefig(output_image_path)

        # Generar imagen para tabla de prueba
        fig_test, ax_test = plt.subplots(figsize=(6, len(test_df) * 0.6))
        ax_test.axis('off')
        table_test = ax_test.table(
            cellText=test_df.values,
            colLabels=test_df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#e49edd', '#e49edd']
        )
        table_test.auto_set_font_size(False)
        table_test.set_fontsize(10)
        table_test.auto_set_column_width(col=list(range(len(test_df.columns))))
        output_image_path = f"{test_filename}.png"
        plt.savefig(output_image_path)

        return fig_train, fig_test

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
        """Guarda las tablas de exactitud en archivos CSV separados"""
        train_filename, test_filename, _, _ = self.generate_file_paths(base_path)
        self.generate_accuracy_tables(train_filename, test_filename)

    def save_feature_importance_plot(self, base_path):
        """Guarda el gráfico de importancia de variables en un archivo"""
        _, _, importance_filename, _ = self.generate_file_paths(base_path)
        if self.feature_importances is not None and self.feature_names is not None:
            fig = self.generate_feature_importance_plot()
            fig.savefig(f"{importance_filename}.png", bbox_inches='tight', dpi=300)
            plt.close(fig)

    def generate_and_save_all(self, base_path):
        """Genera y guarda todas las visualizaciones"""
        self.save_accuracy_tables_to_csv(base_path)
        self.save_feature_importance_plot(base_path)

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
        _, _, _, comparison_filename = self.generate_file_paths(base_path)
        fig = self.generate_comparison_plot()
        fig.savefig(f"{comparison_filename}.png", bbox_inches='tight', dpi=300)
        plt.close(fig)