import os
from fastapi import APIRouter, HTTPException, Query

from backend.services.TextProcessingService import TextProcessingService, SentimentAnalysis
from backend.utils.response_models import CsvRequest

router = APIRouter()

@router.post("/process_csv")
async def process_csv(request: CsvRequest):
    try:
        service = TextProcessingService()
        features_df = service.process_csv(request.csv_path)
        return {"message": "CSV procesado exitosamente", "features": features_df.to_dict('records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/calculate_statistics")
async def calculate_statistics(request: CsvRequest):
    try:
        service = TextProcessingService()
        stats = service.calculate_statistics(request.csv_path)
        return {"message": "Estadísticas calculadas exitosamente", "statistics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/get_correlation')
async def get_correlation(
    csv_path: str = Query(None, description="Csv a procesar")
    ):
    try:
        # Calcular la correlación y devolver la ruta de la imagen
        service = TextProcessingService()
        image_path = service.calculate_correlation(csv_path)
        if os.path.exists(image_path):
            return {"status": "success", "image_path": image_path}
        else:
            raise HTTPException(status_code=500, detail="La imagen de correlación no pudo ser creada.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get('/generate_graphics')
async def get_graphics(
    csv_path: str = Query(None, description="Csv a procesar")
    ):
    try:
        # Generar gráficos para cada columna seleccionada
        service = TextProcessingService()
        service.generate_graphics(csv_path)
        service.generate_frequency_graphics(csv_path)
        return {"status": "success", "message": "Gráficos generados exitosamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/filter_csv")
async def filter_csv(request: CsvRequest):
    try:
        service = TextProcessingService()
        stats = service.filter_data(request.csv_path)
        return {"message": "CSV filtrado exitosamente", "statistics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sentiment_analysis")
async def sentiment_analysis(request: CsvRequest):
    try:
        sentiment_analysis = SentimentAnalysis(request.csv_path)
        sentiment_analysis.plot_histograms()
        output_path = request.csv_path.replace("_filtered.csv", "_sentiment_analysis.csv")
        sentiment_analysis.save_csv(output_path)
        return {
            "message": "CSV filtrado exitosamente",
            "polarity": sentiment_analysis.data['polarity'].tolist(),
            "subjectivity": sentiment_analysis.data['subjectivity'].tolist(),
            "polarity_category": sentiment_analysis.data['polarity_category'].tolist(),
            "subjectivity_category": sentiment_analysis.data['subjectivity_category'].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dump_stopwords")
async def dump_stopwords():
    try:
        service = TextProcessingService()
        stopwords = service.dump_stopwords()
        return stopwords
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))