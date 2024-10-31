from fastapi import APIRouter, HTTPException
from backend.services.TextProcessingService import TextProcessingService
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
    
@router.get("/dump_stopwords")
async def dump_stopwords():
    try:
        service = TextProcessingService()
        stopwords = service.dump_stopwords()
        return stopwords
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))