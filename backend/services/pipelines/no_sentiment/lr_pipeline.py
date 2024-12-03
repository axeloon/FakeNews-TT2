from .base_pipeline import NoSentimentBasePipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from backend.constant import RANDOM_SEED

class LogisticRegressionNoSentimentPipeline(NoSentimentBasePipeline):
    def get_param_grid(self):
        return {
            'lr__C': [1.0],
            'lr__max_iter': [1000],
            'lr__random_state': [RANDOM_SEED]
        }
    
    def create_model(self):
        return Pipeline([
            ('lr', LogisticRegression(random_state=RANDOM_SEED))
        ]) 