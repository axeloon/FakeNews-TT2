import logging
import joblib
import os
import json
import numpy as np
from typing import List
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from backend.utils.response_models import TrainingResponseModel

# Importar pipelines con sentimiento
from backend.services.pipelines.sentiment.svm_pipeline import SVMSentimentPipeline
from backend.services.pipelines.sentiment.lr_pipeline import LogisticRegressionSentimentPipeline
from backend.services.pipelines.sentiment.boosting_pipeline import BoostingSentimentPipeline
from backend.services.pipelines.sentiment.nn_pipeline import NeuralNetworkSentimentPipeline

# Importar pipelines sin sentimiento
from backend.services.pipelines.no_sentiment.svm_pipeline import SVMNoSentimentPipeline
from backend.services.pipelines.no_sentiment.lr_pipeline import LogisticRegressionNoSentimentPipeline
from backend.services.pipelines.no_sentiment.boosting_pipeline import BoostingNoSentimentPipeline
from backend.services.pipelines.no_sentiment.nn_pipeline import NeuralNetworkNoSentimentPipeline
from backend.constant import fine_tuned_dir

logger = logging.getLogger(__name__)

class ModelFineTuner:
    def __init__(self, model_paths: List[str], X_train, X_test, y_train, y_test, with_sentiment: bool = True, pipeline_class = None, name_model: str = None):
        self.model_paths = model_paths
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.with_sentiment = with_sentiment
        self.pipeline_class = pipeline_class
        self.name_model = name_model
        self.pipeline_map = self._get_pipeline_map()
        
        try:
            # Seleccionar el archivo de métricas según el tipo de modelo
            if self.with_sentiment:
                metrics_file = 'backend/data/modelos_originales_sentiment.json'
            else:
                metrics_file = 'backend/data/modelos_originales_no_sentiment.json'
            
            with open(metrics_file, 'r', encoding='utf-8') as f:
                self.original_metrics = {
                    model['name_model']: model 
                    for model in json.load(f)
                }
            logger.info(f"Métricas originales cargadas correctamente para modelos {'con' if self.with_sentiment else 'sin'} sentimiento")
        except Exception as e:
            logger.error(f"Error al cargar métricas originales: {str(e)}")
            raise

    def _get_pipeline_map(self):
        if self.pipeline_class:
            # Si se proporciona un pipeline_class, usarlo para todos los modelos
            return {model_path: (self.pipeline_class, self.name_model) for model_path in self.model_paths}
        
        if self.with_sentiment:
            return {
                'models/svm_ngram_model_sentiment.pkl': 
                    (SVMSentimentPipeline, "N-grams + SVM CS"),
                'models/lr_ngram_model_sentiment.pkl': 
                    (LogisticRegressionSentimentPipeline, "N-grams + Regresión Logística CS"),
                'models/boosting_char_ngram_model_sentiment.pkl': 
                    (BoostingSentimentPipeline, "Boosting (BO) con n-grams de caracteres CS"),
                'models/nn_embedding_model_sentiment.h5': 
                    (NeuralNetworkSentimentPipeline, "Embeddings + Redes Neuronales CS")
            }
        else:
            return {
                'models/svm_ngram_model_no_sentiment.pkl': 
                    (SVMNoSentimentPipeline, "N-grams + SVM SS"),
                'models/lr_ngram_model_no_sentiment.pkl': 
                    (LogisticRegressionNoSentimentPipeline, "N-grams + Regresión Logística SS"),
                'models/boosting_char_ngram_model_no_sentiment.pkl': 
                    (BoostingNoSentimentPipeline, "Boosting (BO) con n-grams de caracteres SS"),
                'models/nn_embedding_model_no_sentiment.h5': 
                    (NeuralNetworkNoSentimentPipeline, "Embeddings + Redes Neuronales SS")
            }

    def fine_tune_models(self) -> List[TrainingResponseModel]:
        logger.info("Iniciando fine-tuning de modelos")
        responses = []
        
        for model_file in self.model_paths:
            try:
                pipeline_class, model_name = self.pipeline_map.get(model_file, (None, None))
                if not pipeline_class:
                    logger.error(f"No se encontró pipeline para el modelo: {model_file}")
                    continue
                
                self.name_model = model_name
                logger.info(f"Iniciando fine-tuning para {model_name}")
                pipeline = pipeline_class(self.X_train, self.X_test, self.y_train, self.y_test)
                fine_tuned_model = pipeline.fine_tune()
                
                # Crear el directorio base si no existe
                os.makedirs(fine_tuned_dir, exist_ok=True)
                # Determinar el subdirectorio basado en la clase del pipeline
                subdir = "best" if pipeline_class and pipeline_class.__name__ == "BoostingPipelineV2" else ""
                # Crear el subdirectorio si es necesario
                if subdir:
                    os.makedirs(os.path.join(fine_tuned_dir, subdir), exist_ok=True)
                # Guardar el modelo fine-tuned
                base_name = os.path.basename(model_file)
                fine_tuned_path = os.path.join(fine_tuned_dir, subdir, base_name)
                joblib.dump(fine_tuned_model, fine_tuned_path)
                logger.info(f"Modelo fine-tuned guardado en {fine_tuned_path}")
                
                # Evaluar el modelo fine-tuned
                metrics = self.evaluate_model(
                    fine_tuned_model, 
                    self.X_train, self.X_test, 
                    self.y_train, self.y_test,
                    model_name
                )
                
                # Obtener métricas originales
                original_metrics = self.original_metrics.get(model_name, {})
                
                # Verificar si el modelo mejoró
                if metrics['f1_score'] > original_metrics.get('f1_score', 0):
                    response = TrainingResponseModel(
                        name_model=model_name,
                        status="Fine-tuning completado",
                        accuracy_train=metrics['accuracy_train'],
                        accuracy=metrics['accuracy'],
                        precision=metrics['precision'],
                        precision_train=metrics['precision_train'],
                        recall=metrics['recall'],
                        recall_train=metrics['recall_train'],
                        f1_score=metrics['f1_score'],
                        f1_score_train=metrics['f1_score_train'],
                        message="El modelo ha mejorado y se ha guardado exitosamente.",
                        feature_importances=original_metrics.get('feature_importances'),
                        feature_names=original_metrics.get('feature_names'),
                        y_true=self.y_test.tolist(),
                        y_pred=metrics['y_pred'].tolist(),
                        y_prob=metrics['y_prob'].tolist() if metrics['y_prob'] is not None else None
                    )
                else:
                    # Si no mejoró, usar las métricas originales
                    response = TrainingResponseModel(
                        name_model=model_name,
                        status="Fine-tuning no mejoró el modelo",
                        accuracy_train=original_metrics['accuracy_train'],
                        accuracy=original_metrics['accuracy'],
                        precision=original_metrics['precision'],
                        precision_train=original_metrics['precision_train'],
                        recall=original_metrics['recall'],
                        recall_train=original_metrics['recall_train'],
                        f1_score=original_metrics['f1_score'],
                        f1_score_train=original_metrics['f1_score_train'],
                        message="Se mantiene el modelo original por mejor rendimiento.",
                        feature_importances=original_metrics.get('feature_importances'),
                        feature_names=original_metrics.get('feature_names'),
                        y_true=self.y_test.tolist(),
                        y_pred=metrics['y_pred'].tolist(),
                        y_prob=metrics['y_prob'].tolist() if metrics['y_prob'] is not None else None
                    )
                
                responses.append(response)
                logger.info(f"Fine-tuning completado para {model_name}")
                
            except Exception as e:
                logger.error(f"Error en fine-tuning del modelo {model_file}: {str(e)}")
                logger.exception("Stacktrace completo:")
                continue
        
        if not responses:
            raise ValueError("No se pudo completar el fine-tuning de ningún modelo")
        
        return responses

    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """Evalúa el modelo y retorna las métricas"""
        try:
            # Para modelos de Keras
            if isinstance(model, (NeuralNetworkNoSentimentPipeline, NeuralNetworkSentimentPipeline)):
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                y_prob_test = model.predict_proba(X_test)
            else:
                # Para modelos sklearn
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                try:
                    y_prob_test = model.predict_proba(X_test)[:, 1]
                except:
                    y_prob_test = model.decision_function(X_test) if hasattr(model, 'decision_function') else None

            # Asegurar que todos los arrays sean 1D
            y_train = np.asarray(y_train).flatten()
            y_test = np.asarray(y_test).flatten()
            y_pred_train = np.asarray(y_pred_train).flatten()
            y_pred_test = np.asarray(y_pred_test).flatten()
            if y_prob_test is not None:
                y_prob_test = np.asarray(y_prob_test).flatten()

            metrics = {
                'accuracy_train': float(accuracy_score(y_train, y_pred_train)),
                'precision_train': float(precision_score(y_train, y_pred_train, average='weighted')),
                'recall_train': float(recall_score(y_train, y_pred_train, average='weighted')),
                'f1_score_train': float(f1_score(y_train, y_pred_train, average='weighted')),
                'accuracy': float(accuracy_score(y_test, y_pred_test)),
                'precision': float(precision_score(y_test, y_pred_test, average='weighted')),
                'recall': float(recall_score(y_test, y_pred_test, average='weighted')),
                'f1_score': float(f1_score(y_test, y_pred_test, average='weighted')),
                'y_true': y_test,
                'y_pred': y_pred_test,
                'y_prob': y_prob_test,
                'name_model': model_name
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error al evaluar el modelo {model_name}: {str(e)}")
            raise
