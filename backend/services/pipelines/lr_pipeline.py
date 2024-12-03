from .base_pipeline import BaseModelPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class LogisticRegressionPipeline(BaseModelPipeline):
    def get_param_grid(self):
        return {
            'lr__C': [0.01, 0.1, 1.0],
            'lr__class_weight': ['balanced', None],
            'lr__solver': ['lbfgs', 'saga'],
            'lr__penalty': ['l2']
        }
    
    def create_model(self):
        return Pipeline([
            ('lr', LogisticRegression(max_iter=5000))
        ]) 