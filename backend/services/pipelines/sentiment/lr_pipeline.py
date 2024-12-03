from .base_pipeline import SentimentBasePipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from backend.constant import RANDOM_SEED

class LogisticRegressionSentimentPipeline(SentimentBasePipeline):
    def get_param_grid(self):
        return {
            'poly__degree': [2],
            'poly__interaction_only': [True],
            'scaler__with_std': [True],
            'feature_selection__n_features_to_select': [10],
            'lr__C': [0.8, 1.0],
            'lr__solver': ['saga'],
            'lr__penalty': ['elasticnet'],
            'lr__l1_ratio': [0.25],
            'lr__class_weight': ['balanced'],
            'lr__max_iter': [25000],
            'lr__random_state': [RANDOM_SEED]
        }
    
    def create_model(self):
        return Pipeline([
            ('poly', PolynomialFeatures(include_bias=False)),
            ('scaler', StandardScaler()),
            ('feature_selection', RFE(
                estimator=LogisticRegression(random_state=RANDOM_SEED),
                n_features_to_select=50,
                step=0.2
            )),
            ('lr', LogisticRegression(random_state=RANDOM_SEED))
        ]) 