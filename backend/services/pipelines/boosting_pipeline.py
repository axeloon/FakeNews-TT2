from .base_pipeline import BaseModelPipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

class BoostingPipeline(BaseModelPipeline):
    def get_param_grid(self):
        return {
            'boosting__n_estimators': [100, 200],
            'boosting__learning_rate': [0.01, 0.05, 0.1],
            'boosting__max_depth': [3, 4],
            'boosting__subsample': [0.8, 0.9],
            'boosting__min_samples_split': [5, 10]
        }
    
    def create_model(self):
        return Pipeline([
            ('boosting', GradientBoostingClassifier())
        ]) 