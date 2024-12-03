from .base_pipeline import NoSentimentBasePipeline
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from backend.constant import RANDOM_SEED

class SVMNoSentimentPipeline(NoSentimentBasePipeline):
    def get_param_grid(self):
        return {
            'svm__C': [1.0],
            'svm__kernel': ['rbf'],
            'svm__class_weight': ['balanced'],
            'svm__random_state': [RANDOM_SEED]
        }
    
    def create_model(self):
        return Pipeline([
            ('svm', SVC(probability=True, random_state=RANDOM_SEED))
        ]) 