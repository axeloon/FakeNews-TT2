from .base_pipeline import BaseModelPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class LogisticRegressionPipeline(BaseModelPipeline):
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
            'lr__max_iter': [25000]
        }
    
    def create_model(self):
        return Pipeline([
            ('poly', PolynomialFeatures(include_bias=False)),
            ('scaler', StandardScaler()),
            ('feature_selection', RFE(
                estimator=LogisticRegression(penalty='l2', random_state=42),
                n_features_to_select=50,
                step=0.2
            )),
            ('lr', LogisticRegression(max_iter=20000, random_state=42))
        ]) 