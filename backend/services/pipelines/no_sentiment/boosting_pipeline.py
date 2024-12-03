from .base_pipeline import NoSentimentBasePipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from backend.constant import RANDOM_SEED

class BoostingNoSentimentPipeline(NoSentimentBasePipeline):
    def get_param_grid(self):
        return {
            'boosting__n_estimators': [100],
            'boosting__random_state': [RANDOM_SEED]
        }
    
    def create_model(self):
        return Pipeline([
            ('boosting', GradientBoostingClassifier(random_state=RANDOM_SEED))
        ]) 