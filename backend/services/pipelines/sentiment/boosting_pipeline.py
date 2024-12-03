from .base_pipeline import SentimentBasePipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from backend.constant import RANDOM_SEED

class BoostingSentimentPipeline(SentimentBasePipeline):
    def get_param_grid(self):
        return {
            'quantile__n_quantiles': [1000],
            'quantile__output_distribution': ['normal'],
            'boosting__n_estimators': [400],
            'boosting__learning_rate': [0.01],
            'boosting__max_depth': [3],
            'boosting__subsample': [0.75],
            'boosting__min_samples_split': [25],
            'boosting__min_samples_leaf': [15],
            'boosting__max_features': [0.7],
            'boosting__validation_fraction': [0.2],
            'boosting__random_state': [RANDOM_SEED]
        }
    
    def create_model(self):
        return Pipeline([
            ('quantile', QuantileTransformer(random_state=RANDOM_SEED)),
            ('boosting', GradientBoostingClassifier(
                random_state=RANDOM_SEED,
                warm_start=True,
                validation_fraction=0.2,
                n_iter_no_change=25,
                tol=1e-4
            ))
        ]) 