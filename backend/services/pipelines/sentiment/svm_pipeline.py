from .base_pipeline import SentimentBasePipeline
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from backend.constant import RANDOM_SEED

class SVMSentimentPipeline(SentimentBasePipeline):
    def get_param_grid(self):
        return {
            'scaler__with_std': [True],
            'svm__C': [0.1, 0.5, 1.0],
            'svm__kernel': ['rbf', 'linear'],
            'svm__gamma': ['scale'],
            'svm__class_weight': ['balanced'],
            'svm__random_state': [RANDOM_SEED]
        }
    
    def create_model(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                probability=True,
                random_state=RANDOM_SEED,
                max_iter=1000
            ))
        ]) 