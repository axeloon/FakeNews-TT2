import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class TrainingResultsVisualization:
    def __init__(self, name_model, accuracy_train, accuracy_test, feature_importances=None, feature_names=None):
        self.name_model = name_model
        self.accuracy_train = accuracy_train
        self.accuracy_test = accuracy_test
        self.feature_importances = feature_importances
        self.feature_names = feature_names

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
        if self.feature_importances is not None and self.feature_names is not None:
            plt.figure(figsize=(10, 6))
            # Tomar solo las top 15 variables más importantes
            indices = np.argsort(self.feature_importances)[-20:][::-1]
            sorted_importances = np.array(self.feature_importances)[indices]
            sorted_features = np.array(self.feature_names)[indices]
            
            plt.barh(sorted_features, sorted_importances, align='center')
            plt.xlabel('Importancia')
            plt.title(f'Top 15 variables más importantes - {self.name_model}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            return plt.gcf()

    def save_accuracy_tables_to_csv(self, train_filename, test_filename):
        """Guarda las tablas de exactitud en archivos CSV separados"""
        train_df, test_df = self.generate_accuracy_tables(train_filename, test_filename)

    def save_feature_importance_plot(self, filename):
        """Guarda el gráfico de importancia de variables en un archivo"""
        if self.feature_importances is not None and self.feature_names is not None:
            fig = self.generate_feature_importance_plot()
            fig.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close(fig)

    def generate_and_save_all(self, accuracy_train_filename, accuracy_test_filename, importance_plot_filename=None):
        """Genera y guarda todas las visualizaciones"""
        # Guardar tablas de exactitud
        self.save_accuracy_tables_to_csv(accuracy_train_filename, accuracy_test_filename)
        
        # Guardar gráfico de importancia si está disponible
        if importance_plot_filename and self.feature_importances is not None:
            self.save_feature_importance_plot(importance_plot_filename)

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

    def save_comparison_plot(self, filename):
        """Guarda el gráfico comparativo en un archivo"""
        fig = self.generate_comparison_plot()
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close(fig)