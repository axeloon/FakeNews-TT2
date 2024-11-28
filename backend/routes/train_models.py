from fastapi import APIRouter, HTTPException
from typing import List
from backend.utils.response_models import TrainingResponseModel
from backend.services.ModelTraining import SentimentAnalysisModelTrainer, NoSentimentAnalysisModelTrainer
from backend.services.TrainingResultsViz import TrainingResultsVisualization

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
            viz.generate_and_save_all(
                f"visualizations/sentiment_{result.name_model}_train_metrics.csv",
                f"visualizations/sentiment_{result.name_model}_test_metrics.csv",
                f"visualizations/sentiment_{result.name_model}_feature_importance.png"
            )
            
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
            viz.generate_and_save_all(
                f"visualizations/no_sentiment{result.name_model}_train_metrics",
                f"visualizations/no_sentiment_{result.name_model}_test_metrics",
                f"visualizations/no_sentiment_{result.name_model}_feature_importance.png"
            )
            
            # Guardar gráfico comparativo
            viz.save_comparison_plot(f"visualizations/no_sentiment_{result.name_model}_comparison.png")

        return training_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
