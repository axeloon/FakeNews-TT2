import logging
import gc
from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from backend.constant import RANDOM_SEED

logger = logging.getLogger(__name__)

class BaseModelPipeline(ABC):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.best_model = None
        
    @abstractmethod
    def get_param_grid(self):
        """Retorna el grid de parámetros específico para el modelo"""
        pass
    
    @abstractmethod
    def create_model(self):
        """Crea y retorna el modelo base"""
        pass
    
    def fine_tune(self):
        """Proceso base de fine-tuning"""
        try:
            self.model = self.create_model()
            param_grid = self.get_param_grid()
            
            cv = StratifiedKFold(
                n_splits=5, 
                shuffle=True, 
                random_state=RANDOM_SEED
            )
            
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=cv,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1,
                refit=True,
                return_train_score=True
            )
            
            grid_search.fit(self.X_train, self.y_train)
            self.best_model = grid_search.best_estimator_
            
            # Limpiar recursos
            gc.collect()
            
            return self.best_model
            
        except Exception as e:
            logger.error(f"Error en fine-tuning: {str(e)}")
            raise