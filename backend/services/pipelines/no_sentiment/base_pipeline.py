import logging
import gc
from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from backend.constant import RANDOM_SEED

logger = logging.getLogger(__name__)

class NoSentimentBasePipeline(ABC):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.best_model = None
        
    @abstractmethod
    def get_param_grid(self):
        """Retorna el grid de parámetros específico para el modelo sin sentimiento"""
        pass
    
    @abstractmethod
    def create_model(self):
        """Crea y retorna el modelo base sin sentimiento"""
        pass
    
    def fine_tune(self):
        """Proceso de fine-tuning para modelos sin sentimiento (mantenido simple)"""
        try:
            self.model = self.create_model()
            self.model.fit(self.X_train, self.y_train)
            self.best_model = self.model
            return self.best_model
            
        except Exception as e:
            logger.error(f"Error en fine-tuning de modelo sin sentimiento: {str(e)}")
            raise 