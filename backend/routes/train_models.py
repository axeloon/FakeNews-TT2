from fastapi import APIRouter, HTTPException
from typing import List
from backend.utils.response_models import TrainingResponseModel
from backend.services.ModelTraining import SentimentAnalysisModelTrainer, NoSentimentAnalysisModelTrainer
from backend.services.ModelFineTuning import SentimentFineTuningModelTrainer, NoSentimentFineTuningModelTrainer
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
                feature_importances=result.feature_importances,
                feature_names=result.feature_names
            )
            
            # Generar y guardar todas las visualizaciones
            viz.generate_and_save_all(f"visualizations/sentiment/{result.name_model}")

            # Guardar gráfico comparativo
            viz.save_comparison_plot(f"visualizations/sentiment_{result.name_model}_comparison.png")

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
                feature_importances=result.feature_importances,
                feature_names=result.feature_names
            )
            
            # Generar y guardar todas las visualizaciones
            viz.generate_and_save_all(f"visualizations/no_sentiment{result.name_model}")

            # Guardar gráfico comparativo
            viz.save_comparison_plot(f"visualizations/no_sentiment_{result.name_model}_comparison.png")

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

        fine_tuner = SentimentFineTuningModelTrainer(model_paths, X_train, X_test, y_train, y_test)
        fine_tuning_results = fine_tuner.fine_tune_models()

        # Crear directorios para visualizaciones
        os.makedirs("visualizations/fine_tuning/sentiment", exist_ok=True)

        # Generar visualizaciones para cada modelo
        for result in fine_tuning_results:
            try:
                # Crear visualización para métricas individuales
                viz = TrainingResultsVisualization(
                    name_model=result.name_model,
                    accuracy_train=result.accuracy_train,
                    accuracy_test=result.accuracy,
                    feature_importances=result.feature_importances,
                    feature_names=result.feature_names
                )
                
                # Generar visualizaciones básicas
                viz.generate_and_save_all(f"visualizations/fine_tuning/sentiment/{result.name_model}")

                # Obtener métricas del modelo original
                original_model = next(
                    (m for m in trainer.train_models() if m.name_model == result.name_model),
                    None
                )

                if original_model:
                    # Generar comparación con modelo original
                    viz.generate_fine_tuning_comparison(
                        original_accuracy_train=original_model.accuracy_train,
                        original_accuracy_test=original_model.accuracy,
                        fine_tuned_accuracy_train=result.accuracy_train,
                        fine_tuned_accuracy_test=result.accuracy,
                        save_path=f"visualizations/fine_tuning/sentiment/{result.name_model}_comparison.png"
                    )
                    
                logger.info(f"Visualizaciones generadas para {result.name_model}")
            except Exception as viz_error:
                logger.error(f"Error generando visualizaciones para {result.name_model}: {str(viz_error)}")
                continue

        TrainingResultsVisualization.generate_fine_tuning_results_table(fine_tuning_results, "visualizations/fine_tuning/sentiment")

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

        fine_tuner = NoSentimentFineTuningModelTrainer(model_paths, X_train, X_test, y_train, y_test)
        fine_tuning_results = fine_tuner.fine_tune_models()

        # Crear directorios para visualizaciones si no existen
        os.makedirs("visualizations/fine_tuning/no_sentiment", exist_ok=True)
        os.makedirs("visualizations/fine_tuning/no_sentiment/comparisons", exist_ok=True)

        # Obtener métricas de modelos originales una sola vez
        original_models_metrics = trainer.train_models()

        # Generar visualizaciones para cada modelo
        for result in fine_tuning_results:
            try:
                original_model_name = result.name_model.replace(" (Fine-Tuned)", "")
                original_metrics = next(
                    (r for r in original_models_metrics if r.name_model == original_model_name),
                    None
                )

                if original_metrics:
                    # Crear visualización para el modelo fine-tuned
                    viz = TrainingResultsVisualization(
                        name_model=result.name_model,
                        accuracy_train=result.accuracy_train,
                        accuracy_test=result.accuracy,
                        feature_importances=result.feature_importances,
                        feature_names=result.feature_names
                    )
                    
                    # Generar todas las visualizaciones estándar
                    viz.generate_and_save_all(f"visualizations/fine_tuning/no_sentiment/{result.name_model.replace(' ', '_')}")
                    
                    # Generar la comparación específica entre original y fine-tuned
                    viz.generate_fine_tuning_comparison(
                        original_accuracy_train=original_metrics.accuracy_train,
                        original_accuracy_test=original_metrics.accuracy,
                        fine_tuned_accuracy_train=result.accuracy_train,
                        fine_tuned_accuracy_test=result.accuracy,
                        save_path=f"visualizations/fine_tuning/no_sentiment/comparisons/{result.name_model.replace(' ', '_')}_comparison.png"
                    )
            except Exception as viz_error:
                logger.error(f"Error al generar visualización para {result.name_model}: {str(viz_error)}")
                continue

        TrainingResultsVisualization.generate_fine_tuning_results_table(fine_tuning_results, "visualizations/fine_tuning/no_sentiment")

        return fine_tuning_results
    except Exception as e:
        logger.error(f"Error en fine-tuning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
