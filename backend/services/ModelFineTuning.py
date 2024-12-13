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

logger = logging.getLogger(__name__)

class ModelFineTuner:
    def __init__(self, model_paths: List[str], X_train, X_test, y_train, y_test, with_sentiment: bool = True):
        self.model_paths = model_paths
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.with_sentiment = with_sentiment
        self.pipeline_map = self._get_pipeline_map()
        self.name_model = None
        
        try:
            metrics_file = 'backend/data/modelos_originales.json'
            with open(metrics_file, 'r', encoding='utf-8') as f:
                self.original_metrics = {
                    model['name_model']: model 
                    for model in json.load(f)
                }
            logger.info("Métricas originales cargadas correctamente")
        except Exception as e:
            logger.error(f"Error al cargar métricas originales: {str(e)}")
            raise

    def _get_pipeline_map(self):
        if self.with_sentiment:
            return {
                'models/svm_ngram_model_sentiment.pkl': 
                    (SVMSentimentPipeline, "N-grams + SVM"),
                'models/lr_ngram_model_sentiment.pkl': 
                    (LogisticRegressionSentimentPipeline, "N-grams + Regresión Logística"),
                'models/boosting_char_ngram_model_sentiment.pkl': 
                    (BoostingSentimentPipeline, "Boosting (BO) con n-grams de caracteres"),
                'models/nn_embedding_model_sentiment.h5': 
                    (NeuralNetworkSentimentPipeline, "Embeddings + Redes Neuronales")
            }
        else:
            return {
                'models/svm_ngram_model_no_sentiment.pkl': 
                    (SVMNoSentimentPipeline, "N-grams + SVM"),
                'models/lr_ngram_model_no_sentiment.pkl': 
                    (LogisticRegressionNoSentimentPipeline, "N-grams + Regresión Logística"),
                'models/boosting_char_ngram_model_no_sentiment.pkl': 
                    (BoostingNoSentimentPipeline, "Boosting (BO) con n-grams de caracteres"),
                'models/nn_embedding_model_no_sentiment.h5': 
                    (NeuralNetworkNoSentimentPipeline, "Embeddings + Redes Neuronales")
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
                        recall=metrics['recall'],
                        f1_score=metrics['f1_score'],
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
                        recall=original_metrics['recall'],
                        f1_score=original_metrics['f1_score'],
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
            if isinstance(model, NeuralNetworkNoSentimentPipeline) or isinstance(model, NeuralNetworkSentimentPipeline):
                y_pred_train = (model.model.predict(X_train) > 0.5).astype(int).ravel()
                y_pred_test = (model.model.predict(X_test) > 0.5).astype(int).ravel()
                y_prob_test = model.model.predict(X_test).ravel()
            else:
                # Para modelos sklearn
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                try:
                    y_prob_test = model.predict_proba(X_test)[:, 1]
                except:
                    y_prob_test = model.decision_function(X_test) if hasattr(model, 'decision_function') else None

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
