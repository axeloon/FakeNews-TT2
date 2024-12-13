from fastapi import APIRouter, HTTPException
from typing import List
from backend.utils.response_models import TrainingResponseModel
from backend.services.ModelTraining import SentimentAnalysisModelTrainer, NoSentimentAnalysisModelTrainer
from backend.services.ModelFineTuning import ModelFineTuner
from backend.services.TrainingResultsViz import TrainingResultsVisualization
import os
import logging

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/train_model/sentiment", response_model=List[TrainingResponseModel])
async def train_model_with_sentiment():
    try:
        trainer = SentimentAnalysisModelTrainer('backend/database/FakeNewsTT2.db', 'noticias_redes_sociales_sentiment_analysis')
        training_results = trainer.train_models()

        # Crear visualizaciones para cada modelo
        for result in training_results:
            viz = TrainingResultsVisualization(
                name_model=result.name_model,
                accuracy_train=result.accuracy_train,
                accuracy_test=result.accuracy,
                precision=result.precision,
                recall=result.recall,
                f1_score=result.f1_score,
                y_true=result.y_true,
                y_pred=result.y_pred,
                y_prob=result.y_prob,
                feature_importances=result.feature_importances,
                feature_names=result.feature_names
            )
            

        # Generar tabla de resultados general
        TrainingResultsVisualization.generate_fine_tuning_results_table(
            training_results, 
            "visualizations/sentiment"
        )

        return training_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train_model/no_sentiment", response_model=List[TrainingResponseModel])
async def train_model_without_sentiment():
    try:
        trainer = NoSentimentAnalysisModelTrainer('backend/database/FakeNewsTT2.db', 'noticias_redes_sociales_filtered')
        training_results = trainer.train_models()

        # Crear visualizaciones para cada modelo
        for result in training_results:
            viz = TrainingResultsVisualization(
                name_model=result.name_model,
                accuracy_train=result.accuracy_train,
                accuracy_test=result.accuracy,
                precision=result.precision,
                recall=result.recall,
                f1_score=result.f1_score,
                y_true=result.y_true,
                y_pred=result.y_pred,
                y_prob=result.y_prob,
                feature_importances=result.feature_importances,
                feature_names=result.feature_names
            )
            
        # Generar tabla de resultados general
        TrainingResultsVisualization.generate_fine_tuning_results_table(
            training_results, 
            "visualizations/no_sentiment"
        )

        return training_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fine_tune/sentiment", response_model=List[TrainingResponseModel])
async def fine_tune_model_with_sentiment():
    try:
        trainer = SentimentAnalysisModelTrainer('backend/database/FakeNewsTT2.db', 'noticias_redes_sociales_sentiment_analysis')
        X_train, X_test = trainer.X_train, trainer.X_test
        y_train, y_test = trainer.y_train, trainer.y_test

        model_paths = [
            'models/svm_ngram_model_sentiment.pkl',
            'models/lr_ngram_model_sentiment.pkl',
            'models/boosting_char_ngram_model_sentiment.pkl',
            'models/nn_embedding_model_sentiment.h5'
        ]

        fine_tuner = ModelFineTuner(
            model_paths, 
            X_train, X_test, 
            y_train, y_test, 
            with_sentiment=True
        )
        fine_tuning_results = fine_tuner.fine_tune_models()

        # Generar visualizaciones usando el método unificado
        TrainingResultsVisualization.generate_fine_tuning_results_table(
            fine_tuning_results, 
            os.path.join("visualizations", "fine_tuning", "sentiment"),
            with_sentiment=True
        )

        return fine_tuning_results
    except Exception as e:
        logger.error(f"Error en fine-tuning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fine_tune/no_sentiment", response_model=List[TrainingResponseModel])
async def fine_tune_model_without_sentiment():
    try:
        trainer = NoSentimentAnalysisModelTrainer('backend/database/FakeNewsTT2.db', 'noticias_redes_sociales_filtered')
        X_train, X_test = trainer.X_train, trainer.X_test
        y_train, y_test = trainer.y_train, trainer.y_test

        model_paths = [
            'models/svm_ngram_model_no_sentiment.pkl',
            'models/lr_ngram_model_no_sentiment.pkl',
            'models/boosting_char_ngram_model_no_sentiment.pkl',
            'models/nn_embedding_model_no_sentiment.h5'
        ]

        fine_tuner = ModelFineTuner(
            model_paths, 
            X_train, X_test, 
            y_train, y_test, 
            with_sentiment=False
        )
        fine_tuning_results = fine_tuner.fine_tune_models()

        # Generar visualizaciones usando el método unificado
        TrainingResultsVisualization.generate_fine_tuning_results_table(
            fine_tuning_results, 
            os.path.join("visualizations", "fine_tuning", "no_sentiment"),
            with_sentiment=False
        )

        return fine_tuning_results
    except Exception as e:
        logger.error(f"Error en fine-tuning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
