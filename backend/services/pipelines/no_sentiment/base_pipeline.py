import logging
import gc
from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from backend.constant import RANDOM_SEED
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import numpy as np

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
            
            # Si es un modelo de Keras, ya tiene su propio proceso de fine-tuning
            if isinstance(self.model, (Sequential, KerasClassifier)):
                self.best_model = self.model
                # Agregar clases para compatibilidad con scikit-learn
                if isinstance(self.best_model, KerasClassifier):
                    self.best_model.classes_ = np.unique(self.y_train)
            else:
                # Proceso normal para modelos sklearn
                self.model.fit(self.X_train, self.y_train)
                self.best_model = self.model
            
            return self.best_model
            
        except Exception as e:
            logger.error(f"Error en fine-tuning: {str(e)}")
            raise