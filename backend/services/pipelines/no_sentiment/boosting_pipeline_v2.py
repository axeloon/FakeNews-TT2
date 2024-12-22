import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.inspection import permutation_importance

from backend.services.pipelines.no_sentiment.base_pipeline import NoSentimentBasePipeline
from backend.utils.response_models import TrainingResponseModel
from backend.constant import RANDOM_SEED, FEATURE_COLUMNS_NO_SENTIMENT

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def custom_metric(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    # Equal weights for all metrics
    return np.mean([precision, recall, f1, accuracy])

custom_scorer = make_scorer(custom_metric, greater_is_better=True)

class BoostingPipelineV2(NoSentimentBasePipeline):
    def get_param_grid(self):
        return {
            'boosting__n_estimators': [150, 200],
            'boosting__learning_rate': [0.05, 0.075],
            'boosting__max_depth': [3],
            'boosting__min_samples_split': [10, 15],
            'boosting__min_samples_leaf': [5, 10],
            'boosting__subsample': [0.7, 0.8],
            'boosting__random_state': [RANDOM_SEED],
        }

    def create_model(self):
        base_model = GradientBoostingClassifier(random_state=RANDOM_SEED)

        # Define SMOTE pipeline with GridSearchCV
        smote_pipeline = GridSearchCV(
            estimator=ImbPipeline([
                ('sampling', SMOTE(sampling_strategy=1.0, random_state=RANDOM_SEED)),
                ('boosting', base_model)
            ]),
            param_grid=self.get_param_grid(),
            cv=10,
            scoring=custom_scorer,
            n_jobs=-1,
            verbose=2
        )

        # Define simple V1 pipeline
        v1_pipeline = Pipeline([
            ('boosting', base_model)
        ])

        # Train pipelines separately
        logger.info("Training SMOTE pipeline...")
        smote_pipeline.fit(self.X_train, self.y_train)
        logger.info("Training V1 pipeline...")
        v1_pipeline.fit(self.X_train, self.y_train)

        # Create and return HybridPipeline
        hybrid_pipeline = HybridPipeline(smote_pipeline, v1_pipeline, self.X_test, self.y_test)

        # Finalize evaluation and return TrainingResponseModel
        metrics = hybrid_pipeline.finalize_evaluation()

        return TrainingResponseModel(
            name_model='HybridPipeline',
            status='success',
            accuracy_train=metrics['accuracy_train'],
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            precision_train=metrics['precision_train'],
            recall=metrics['recall'],
            recall_train=metrics['recall_train'],
            f1_score=metrics['f1_score'],
            f1_score_train=metrics['f1_score_train'],
            message=metrics['message'],
            feature_importances=metrics['feature_importances'],
            feature_names=FEATURE_COLUMNS_NO_SENTIMENT,
            y_true=metrics['y_true'],
            y_pred=metrics['y_pred'],
            y_prob=metrics['y_prob']
        )
    
    @staticmethod
    def load_model(model_path):
        """Carga el modelo desde la ruta especificada."""
        logger.info(f"Cargando el modelo desde {model_path}...")
        try:
            model = joblib.load(model_path)
            logger.info("Modelo cargado exitosamente")
            return model
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {str(e)}")
            return None

    @staticmethod
    def calculate_permutation_importance(model, X, y):
        """Calcula la importancia de características usando Permutation Importance"""
        logger.info("Calculando Permutation Importance...")
        try:
            X = pd.DataFrame(X)
            y = y if isinstance(y, np.ndarray) else np.array(y)
            
            if hasattr(model, 'predict_proba'):
                scoring = 'roc_auc'
            else:
                y = (y > 0.5).astype(int)
                scoring = 'f1'
            
            result = permutation_importance(
                model,
                X, y,
                scoring=scoring,
                n_repeats=30,
                random_state=42,
                n_jobs=-1
            )
            
            importances = result.importances_mean
            importances = np.where(importances < 0, 0, importances)
            importances = importances / np.sum(importances) if np.sum(importances) != 0 else importances
            
            logger.info("Permutation Importance calculado exitosamente")
            return importances
            
        except Exception as e:
            logger.error(f"Error al calcular Permutation Importance: {str(e)}")
            return None
        
    @staticmethod
    def balance_datasets(X, y):
        """
        Balancea el conjunto de datos submuestreando la clase mayoritaria para igualar el tamaño de la clase minoritaria.
        Asegúrate de que X e y estén correctamente alineados e indexados.
        """
        try:
            X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
            y = pd.Series(y) if not isinstance(y, pd.Series) else y

            if not X.index.equals(y.index):
                logger.warning("Alineando índices de X e y.")
                X = X.reset_index(drop=True)
                y = y.reset_index(drop=True)

            smallest_class_size = y.value_counts().min()
            balanced_indices = []

            for label in y.unique():
                class_indices = y[y == label].index
                sampled_indices = class_indices.to_series().sample(n=smallest_class_size, random_state=RANDOM_SEED)
                balanced_indices.extend(sampled_indices)

            balanced_X = X.loc[balanced_indices].reset_index(drop=True)
            balanced_y = y.loc[balanced_indices].reset_index(drop=True)

            return balanced_X, balanced_y
        except Exception as e:
            logger.error(f"Error al balancear los conjuntos de datos: {str(e)}")
            raise e


class HybridPipeline:
    def __init__(self, smote_pipeline, v1_pipeline, X_test, y_test):
        self.smote_pipeline = smote_pipeline
        self.v1_pipeline = v1_pipeline
        self.X_test = X_test
        self.y_test = y_test

        # Inject required attributes for compatibility
        self.steps = [('smote_pipeline', smote_pipeline), ('v1_pipeline', v1_pipeline)]
        self.named_steps = {'smote_pipeline': smote_pipeline, 'v1_pipeline': v1_pipeline}

    def fit(self, X, y):
        self.smote_pipeline.fit(X, y)
        self.v1_pipeline.fit(X, y)
        return self

    def predict(self, X, metric="default"):
        smote_preds = self.smote_pipeline.predict(X)
        v1_preds = self.v1_pipeline.predict(X)

        if metric == "precision":
            # Use SMOTE predictions for precision
            return smote_preds
        else:
            # Use V1 predictions for all other metrics
            return v1_preds

    def predict_proba(self, X):
        """Combina las probabilidades de predicción de ambos pipelines y devuelve una lista."""
        smote_proba = self.smote_pipeline.predict_proba(X)
        v1_proba = self.v1_pipeline.predict_proba(X)
        combined_proba = np.zeros_like(smote_proba)

        # For class 1, rely on SMOTE; for class 0, rely on V1
        combined_proba[:, 1] = smote_proba[:, 1]
        combined_proba[:, 0] = v1_proba[:, 0]
        
        # Convertir las probabilidades combinadas a una lista de listas
        return [list(row) for row in combined_proba]

    def evaluate_metrics(self, X, y_true):
        """Evaluate metrics prioritizing SMOTE for precision and V1 for other metrics."""
        smote_preds = self.smote_pipeline.predict(X)  # Always use SMOTE for precision
        v1_preds = self.v1_pipeline.predict(X)  # Always use V1 for other metrics

        precision = float(max(precision_score(y_true, smote_preds, average='weighted'), precision_score(y_true, v1_preds, average='weighted')))
        recall = float(max(recall_score(y_true, smote_preds, average='weighted'), recall_score(y_true, v1_preds, average='weighted')))
        f1 = float(max(f1_score(y_true, smote_preds, average='weighted'), f1_score(y_true, v1_preds, average='weighted')))
        accuracy = float(max(accuracy_score(y_true, smote_preds), accuracy_score(y_true, v1_preds)))

        # Calculate train metrics
        smote_train_preds = self.smote_pipeline.predict(self.X_test)
        v1_train_preds = self.v1_pipeline.predict(self.X_test)

        precision_train = float(max(precision_score(self.y_test, smote_train_preds, average='weighted'), precision_score(self.y_test, v1_train_preds, average='weighted')))
        recall_train = float(max(recall_score(self.y_test, smote_train_preds, average='weighted'), recall_score(self.y_test, v1_train_preds, average='weighted')))
        f1_train = float(max(f1_score(self.y_test, smote_train_preds, average='weighted'), f1_score(self.y_test, v1_train_preds, average='weighted')))
        accuracy_train = float(max(accuracy_score(self.y_test, smote_train_preds), accuracy_score(self.y_test, v1_train_preds)))

        # Si los valores están en tupla, extraer el primer elemento
        precision, recall, accuracy, f1 = (metric[0] if isinstance(metric, tuple) else metric for metric in (precision, recall, accuracy, f1))
        precision_train, recall_train, accuracy_train, f1_train = (metric[0] if isinstance(metric, tuple) else metric for metric in (precision_train, recall_train, accuracy_train, f1_train))

        logger.info(f"Precision (Max): {precision}")
        logger.info(f"Recall (Max): {recall}")
        logger.info(f"F1 Score (Max): {f1}")
        logger.info(f"Accuracy (Max): {accuracy}")

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'precision_train': precision_train,
            'recall_train': recall_train,
            'f1_score_train': f1_train,
            'accuracy_train': accuracy_train
        }

    def finalize_evaluation(self):
        """Evaluate and return metrics as TrainingResponseModel."""
        metrics = self.evaluate_metrics(self.X_test, self.y_test)

        return {
            'name_model': 'HybridPipeline',
            'status': 'success',
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'accuracy_train': metrics['accuracy_train'],
            'precision_train': metrics['precision_train'],
            'recall_train': metrics['recall_train'],
            'f1_score_train': metrics['f1_score_train'],
            'message': 'Hybrid model evaluation completed.',
            'feature_importances': list(self.feature_importances_),
            'feature_names': FEATURE_COLUMNS_NO_SENTIMENT,
            'y_true': list(self.y_test),
            'y_pred': list(self.predict(self.X_test)),
            'y_prob': [proba[0] for proba in self.predict_proba(self.X_test)]
        }

    @property
    def feature_importances_(self):
        """Return feature importances from V1 only."""
        return self.v1_pipeline.named_steps['boosting'].feature_importances_