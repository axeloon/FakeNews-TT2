import logging
import joblib
import os
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
from abc import ABC, abstractmethod
from typing import List
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from backend.utils.response_models import TrainingResponseModel
from sklearn.inspection import permutation_importance
from backend.constant import FEATURE_COLUMNS
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FineTuningModelTrainer(ABC):
    def __init__(self, model_paths: List[str], X_train, X_test, y_train, y_test):
        """
        Clase base abstracta para realizar fine-tuning en modelos ya entrenados.
        :param model_paths: Lista de rutas de los modelos ya entrenados.
        :param X_train: Datos de entrenamiento.
        :param X_test: Datos de prueba.
        :param y_train: Etiquetas de entrenamiento.
        :param y_test: Etiquetas de prueba.
        """
        self.model_paths = model_paths
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.fine_tuned_models = []

    @abstractmethod
    def fine_tune_models(self) -> List[TrainingResponseModel]:
        pass

    def load_model(self, model_path):
        """Carga un modelo dado el path."""
        try:
            if model_path.endswith('.pkl'):
                return joblib.load(model_path)
            elif model_path.endswith('.h5'):
                return KerasClassifier(model=load_model(model_path))
            else:
                raise ValueError(f"Formato de modelo no soportado: {model_path}")
        except Exception as e:
            logger.error(f"Error al cargar el modelo {model_path}: {str(e)}")
            raise

    def save_model(self, model, filename):
        """Guarda el modelo mejorado."""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            if filename.endswith('.pkl'):
                joblib.dump(model, filename)
            elif filename.endswith('.h5'):
                if hasattr(model, 'model'):
                    model.model.save(filename)
                else:
                    model.save(filename)
            logger.info(f"Modelo guardado exitosamente en {filename}")
        except Exception as e:
            logger.error(f"Error al guardar el modelo {filename}: {str(e)}")
            raise

    def get_feature_importances(self, model):
        """Extrae importancia de características usando el método más apropiado"""
        logger.info("Extrayendo importancia de características...")
        
        try:
            # Para modelos que tienen feature_importances_ nativo
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_, FEATURE_COLUMNS
            
            # Para modelos que tienen coef_ (como Logistic Regression)
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
                return importances, FEATURE_COLUMNS
            
            # Para SVM y Redes Neuronales, usar Permutation Importance
            else:
                return self.calculate_permutation_importance(model, self.X_test, self.y_test)
                
        except Exception as e:
            logger.error(f"Error al extraer importancias: {str(e)}")
            return None, None

    def calculate_permutation_importance(self, model, X, y):
        """Calcula la importancia de características usando Permutation Importance"""
        logger.info("Calculando Permutation Importance...")
        try:
            # Asegurarse de que los datos estén en el formato correcto
            X = pd.DataFrame(X, columns=FEATURE_COLUMNS)
            y = y if isinstance(y, np.ndarray) else np.array(y)
            
            # Asegurarse de que el modelo tenga un método predict adecuado
            if not hasattr(model, 'predict'):
                raise ValueError("El modelo no tiene un método predict válido para calcular la importancia de características.")
            
            # Convertir predicciones continuas a binarias si es necesario
            if hasattr(model, 'predict_proba'):
                scoring = 'roc_auc'  # Utilizar AUC si el modelo soporta predict_proba
            else:
                scoring = 'f1'  # Cambiar a F1 para mayor sensibilidad
            
            # Calcular permutation importance
            result = permutation_importance(
                model,
                X, y,
                scoring=scoring,  # Especificamos explícitamente el scoring
                n_repeats=30,  # Aumentar el número de repeticiones para mayor estabilidad
                random_state=42,
                n_jobs=-1
            )
            
            importances = result.importances_mean
            importances = np.where(importances < 0, 0, importances)  # Establecer valores negativos en 0
            importances = importances / np.sum(importances) if np.sum(importances) != 0 else importances
            feature_names = FEATURE_COLUMNS
            
            logger.info("Permutation Importance calculado exitosamente")
            return importances, feature_names
            
        except Exception as e:
            logger.error(f"Error al calcular Permutation Importance: {str(e)}")
            return None, None

class SentimentFineTuningModelTrainer(FineTuningModelTrainer):
    def fine_tune_models(self) -> List[TrainingResponseModel]:
        logger.info("Iniciando fine-tuning de modelos con análisis de sentimiento")
        results = []

        for model_path in self.model_paths:
            try:
                model = self.load_model(model_path)
                model_name = os.path.basename(model_path).split('.')[0]
                logger.info(f"Fine-tuning del modelo: {model_name}")

                feature_importances = None
                feature_names = None

                if isinstance(model, Pipeline):
                    best_model = self._fine_tune_pipeline(model, model_name)
                    
                    # Extraer feature importances según el tipo de modelo
                    if 'svm' in model.named_steps:
                        # Para SVM usamos get_feature_importances que llamará a calculate_permutation_importance
                        feature_importances, feature_names = self.get_feature_importances(best_model.named_steps['svm'])
                        
                    elif 'lr' in model.named_steps:
                        feature_importances, feature_names = self.get_feature_importances(best_model.named_steps['lr']) 
                        
                    elif 'boosting' in model.named_steps:
                        feature_importances, feature_names = self.get_feature_importances(best_model.named_steps['boosting'])
                        
                elif model_path.endswith('.h5'):
                    best_model = self._fine_tune_neural_network(model)
                    feature_importances, feature_names = self.get_feature_importances(best_model)
                    
                else:
                    logger.warning(f"Tipo de modelo no soportado para fine-tuning: {model_name}")
                    continue

                metrics = self.evaluate_model(best_model)
                new_filename = f'models/fine_tuned/{model_name}_fine_tuned{"_sentiment" if "sentiment" in model_path else ""}.{model_path.split(".")[-1]}'
                self.save_model(best_model, new_filename)

                # Crear el resultado con el mismo formato que el entrenamiento original
                result = TrainingResponseModel(
                    name_model=f"{model_name} (Fine-Tuned)",  # Agregar indicador de fine-tuning al nombre
                    status="Fine-tuning completado",
                    accuracy_train=metrics['accuracy_train'],
                    accuracy=metrics['accuracy_test'],
                    f1_score=metrics['f1'],
                    precision=metrics['precision'],
                    recall=metrics['recall'],
                    message="El modelo se ha mejorado y guardado exitosamente.",
                    feature_importances=feature_importances.tolist() if feature_importances is not None else None,
                    feature_names=feature_names if feature_names is not None else None
                )
                
                results.append(result)
                logger.info(f"Fine-tuning completado exitosamente para {model_name}")

            except Exception as e:
                logger.error(f"Error en fine-tuning del modelo {model_path}: {str(e)}")
                continue

        logger.info("Fine-tuning de todos los modelos completado exitosamente")
        return results

    def _fine_tune_pipeline(self, model, model_name):
        """Realiza fine-tuning específico para modelos Pipeline"""
        if 'svm' in model.named_steps:
            param_grid = {'svm__C': [0.1, 1, 10, 50, 100], 'svm__kernel': ['linear', 'rbf', 'poly']}
        elif 'lr' in model.named_steps:
            param_grid = {'lr__C': [0.01, 0.1, 1, 10, 100], 'lr__max_iter': [1000, 1500, 2000]}
        elif 'boosting' in model.named_steps:
            param_grid = {
                'boosting__n_estimators': [50, 100, 200],  # Reducir el número de estimadores para evitar sobreajuste
                'boosting__learning_rate': [0.01, 0.05, 0.1],  # Reducir learning rate para evitar sobreajuste
                'boosting__max_depth': [3, 4],  # Reducir la profundidad máxima para mejorar la generalización
                'boosting__min_samples_split': [20, 30],  # Aumentar min_samples_split para regularización
                'boosting__min_samples_leaf': [8, 10],  # Aumentar min_samples_leaf para regularización
                'boosting__subsample': [0.7, 0.8, 0.9]  # Añadir subsampling para reducir el sobreajuste
            }
        else:
            raise ValueError(f"No se encontró un paso reconocible para fine-tuning en el modelo: {model_name}")

        grid_search = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=10), n_jobs=-1, verbose=2, scoring='f1_weighted')
        grid_search.fit(self.X_train, self.y_train)
        return grid_search.best_estimator_

    def _fine_tune_neural_network(self, model):
        """Realiza fine-tuning específico para redes neuronales con Early Stopping"""
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(self.X_train, self.y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
        return model

    def evaluate_model(self, model):
        """Evalúa el modelo en conjuntos de entrenamiento y prueba."""
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)

        # Asegurarse de que las predicciones sean de tipo binario si es necesario
        if y_pred_train.ndim > 1:
            y_pred_train = np.argmax(y_pred_train, axis=1)
        if y_pred_test.ndim > 1:
            y_pred_test = np.argmax(y_pred_test, axis=1)

        metrics = {
            'accuracy_train': accuracy_score(self.y_train, y_pred_train),
            'accuracy_test': accuracy_score(self.y_test, y_pred_test),
            'f1': f1_score(self.y_test, y_pred_test, average='weighted'),
            'precision': precision_score(self.y_test, y_pred_test, average='weighted', zero_division=1),
            'recall': recall_score(self.y_test, y_pred_test, average='weighted', zero_division=1)
        }

        return metrics

class NoSentimentFineTuningModelTrainer(SentimentFineTuningModelTrainer):
    """
    Hereda todos los métodos de SentimentFineTuningModelTrainer ya que la lógica es la misma,
    solo cambia el contexto de uso (modelos sin análisis de sentimiento)
    """
    pass
