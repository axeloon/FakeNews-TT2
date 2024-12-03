import logging
import joblib
import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from abc import ABC, abstractmethod
from typing import List
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from backend.utils.response_models import TrainingResponseModel
from sklearn.inspection import permutation_importance
from backend.constant import FEATURE_COLUMNS

from backend.services.pipelines.svm_pipeline import SVMPipeline
from backend.services.pipelines.lr_pipeline import LogisticRegressionPipeline
from backend.services.pipelines.boosting_pipeline import BoostingPipeline
from backend.services.pipelines.nn_pipeline import NeuralNetworkPipeline

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
    def __init__(self, model_paths: List[str], X_train, X_test, y_train, y_test):
        super().__init__(model_paths, X_train, X_test, y_train, y_test)
        try:
            # Cargar métricas originales
            with open('backend/data/modelos_originales.json', 'r', encoding='utf-8') as f:
                self.original_metrics = {
                    model['name_model']: model 
                    for model in json.load(f)
                }
            logger.info("Métricas originales cargadas correctamente")
        except Exception as e:
            logger.error(f"Error al cargar métricas originales: {str(e)}")
            raise

    def fine_tune_models(self) -> List[TrainingResponseModel]:
        logger.info("Iniciando fine-tuning de modelos")
        responses = []
        
        pipeline_map = {
            'models/svm_ngram_model_sentiment.pkl': 
                (SVMPipeline, "N-grams + SVM"),
            'models/lr_ngram_model_sentiment.pkl': 
                (LogisticRegressionPipeline, "N-grams + Regresión Logística"),
            'models/boosting_char_ngram_model_sentiment.pkl': 
                (BoostingPipeline, "Boosting (BO) con n-grams de caracteres"),
            'models/nn_embedding_model_sentiment.h5': 
                (NeuralNetworkPipeline, "Embeddings + Redes Neuronales")
        }

        for model_file in self.model_paths:
            try:
                pipeline_class, model_name = pipeline_map.get(model_file, (None, None))
                if not pipeline_class:
                    logger.error(f"No se encontró pipeline para el modelo: {model_file}")
                    continue
                
                logger.info(f"Iniciando fine-tuning para {model_name}")
                pipeline = pipeline_class(self.X_train, self.X_test, self.y_train, self.y_test)
                fine_tuned_model = pipeline.fine_tune()
                
                # Evaluar el modelo
                y_pred_train = fine_tuned_model.predict(self.X_train)
                y_pred_test = fine_tuned_model.predict(self.X_test)
                
                # Asegurarse de que las predicciones sean binarias
                if len(y_pred_train.shape) > 1:
                    y_pred_train = (y_pred_train > 0.5).astype(int)
                if len(y_pred_test.shape) > 1:
                    y_pred_test = (y_pred_test > 0.5).astype(int)
                
                metrics = {
                    'accuracy_train': accuracy_score(self.y_train, y_pred_train),
                    'accuracy_test': accuracy_score(self.y_test, y_pred_test),
                    'f1': f1_score(self.y_test, y_pred_test, average='weighted'),
                    'precision': precision_score(self.y_test, y_pred_test, average='weighted'),
                    'recall': recall_score(self.y_test, y_pred_test, average='weighted')
                }
                
                # Obtener métricas originales
                original_metrics = self.original_metrics.get(model_name, {})
                
                # Verificar si el modelo mejoró
                if metrics['f1'] > original_metrics.get('f1_score', 0):
                    # Guardar modelo fine-tuned
                    fine_tuned_path = os.path.join(
                        "backend", "models", "fine_tuned",
                        f"{os.path.splitext(os.path.basename(model_file))[0]}_fine_tuned{os.path.splitext(model_file)[1]}"
                    )
                    os.makedirs(os.path.dirname(fine_tuned_path), exist_ok=True)
                    
                    if model_file.endswith('.h5'):
                        fine_tuned_model.save(fine_tuned_path)
                    else:
                        joblib.dump(fine_tuned_model, fine_tuned_path)
                    
                    # Calcular importancia de características
                    if hasattr(fine_tuned_model, 'feature_importances_'):
                        feature_importances = fine_tuned_model.feature_importances_
                        feature_names = FEATURE_COLUMNS
                    else:
                        feature_importances, feature_names = self.calculate_permutation_importance(
                            fine_tuned_model, self.X_test, self.y_test
                        )
                    
                    response = TrainingResponseModel(
                        name_model=f"{model_name} (Fine-Tuned)",
                        status="Fine-tuning completado",
                        accuracy=metrics['accuracy_test'],
                        accuracy_train=metrics['accuracy_train'],
                        f1_score=metrics['f1'],
                        precision=metrics['precision'],
                        recall=metrics['recall'],
                        message="El modelo ha mejorado y se ha guardado exitosamente.",
                        feature_importances=feature_importances.tolist() if feature_importances is not None else None,
                        feature_names=feature_names if feature_names is not None else None
                    )
                else:
                    response = TrainingResponseModel(
                        name_model=model_name,
                        status="Fine-tuning no mejoró el modelo",
                        accuracy=original_metrics['accuracy'],
                        accuracy_train=original_metrics['accuracy_train'],
                        f1_score=original_metrics['f1_score'],
                        precision=original_metrics['precision'],
                        recall=original_metrics['recall'],
                        message="Se mantiene el modelo original por mejor rendimiento.",
                        feature_importances=original_metrics['feature_importances'],
                        feature_names=original_metrics['feature_names']
                    )
                
                responses.append(response)
                logger.info(f"Fine-tuning completado para {model_name}")
                
            except Exception as e:
                logger.error(f"Error en fine-tuning del modelo {model_file}: {str(e)}")
                logger.exception("Stacktrace completo:")
                continue
        
        if not responses:
            logger.warning("No se pudo completar el fine-tuning de ningún modelo")
        else:
            logger.info(f"Fine-tuning completado para {len(responses)} modelos")
        
        return responses

class NoSentimentFineTuningModelTrainer(SentimentFineTuningModelTrainer):
    """
    Hereda todos los métodos de SentimentFineTuningModelTrainer ya que la lógica es la misma,
    solo cambia el contexto de uso (modelos sin análisis de sentimiento)
    """
    pass
