from .base_pipeline import BaseModelPipeline
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

class SVMPipeline(BaseModelPipeline):
    def get_param_grid(self):
        return {
            'svm__C': [0.1, 0.5, 1.0, 2.0],
            'svm__kernel': ['rbf', 'linear'],
            'svm__class_weight': ['balanced', None]
        }
    
    def create_model(self):
        return Pipeline([
            ('svm', SVC(probability=True))
        ]) 