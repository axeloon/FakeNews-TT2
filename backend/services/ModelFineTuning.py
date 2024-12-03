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
                        fine_tuned_model.model.save(fine_tuned_path)
                    else:
                        joblib.dump(fine_tuned_model, fine_tuned_path)
                    
                    # Calcular importancia de características
                    feature_importances = getattr(fine_tuned_model, 'feature_importances_', None)
                    feature_names = None
                    
                    if feature_importances is not None:
                        feature_names = list(range(len(feature_importances)))
                    
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
                        feature_names=feature_names
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
                        feature_importances=original_metrics.get('feature_importances'),
                        feature_names=original_metrics.get('feature_names')
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
