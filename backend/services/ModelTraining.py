import pandas as pd
import numpy as np
import logging
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, StandardScaler
from abc import ABC, abstractmethod
from typing import List
from sklearn.inspection import permutation_importance
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from backend.database.connection import SessionLocal
from backend.utils.response_models import TrainingResponseModel
from backend.constant import (
    FEATURE_COLUMNS_SENTIMENT, 
    FEATURE_COLUMNS_NO_SENTIMENT,
    RANDOM_SEED
)

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_required_directories():
    """Crea los directorios necesarios para el almacenamiento de modelos y visualizaciones"""
    directories = ['models', 'visualizations']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Carpeta '{directory}' creada exitosamente")

create_required_directories()

class ModelTrainer(ABC):
    def __init__(self, db_path, table_name):
        logger.info(f"Inicializando ModelTrainer con db_path: {db_path} y table_name: {table_name}")
        self.db_path = db_path
        self.table_name = table_name
        self.data = self.load_data()
        logger.info(f"Datos cargados exitosamente. Tamaño del dataset: {len(self.data)}")
        
        # Seleccionar columnas de características según el nombre de la tabla
        if 'sentiment_analysis' in self.table_name:
            self.feature_columns = FEATURE_COLUMNS_SENTIMENT
        else:
            self.feature_columns = FEATURE_COLUMNS_NO_SENTIMENT
        
        self.X = self.data[self.feature_columns]
        self.y = self.data['is_false']
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(self.y)
        
        # Escalar los datos
        scaler = StandardScaler()
        self.X = pd.DataFrame(scaler.fit_transform(self.X), columns=self.feature_columns)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=RANDOM_SEED
        )
        logger.info(f"División de datos completada. Conjunto de entrenamiento: {len(self.X_train)}, Conjunto de prueba: {len(self.X_test)}")
        
        self.name_model = None
        self.metrics = {}

    def load_data(self):
        logger.info("Cargando datos desde la base de datos...")
        db = SessionLocal()
        query = f"SELECT * FROM {self.table_name}"
        data = pd.read_sql_query(query, db.bind)
        db.close()
        logger.info("Datos cargados exitosamente")
        return data

    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """Evalúa el modelo en conjuntos de entrenamiento y prueba"""
        logger.info(f"Evaluando modelo {self.name_model}...")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_prob_test = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = {
            'accuracy_train': accuracy_score(y_train, y_pred_train),
            'accuracy_test': accuracy_score(y_test, y_pred_test),
            'f1_train': f1_score(y_train, y_pred_train, average='weighted'),
            'f1': f1_score(y_test, y_pred_test, average='weighted'),
            'precision_train': precision_score(y_train, y_pred_train, average='weighted'),
            'precision': precision_score(y_test, y_pred_test, average='weighted'),
            'recall_train': recall_score(y_train, y_pred_train, average='weighted'),
            'recall': recall_score(y_test, y_pred_test, average='weighted'),
            'y_true': y_test,
            'y_pred': y_pred_test,
            'y_prob': y_prob_test
        }
        
        logger.info(f"Métricas de evaluación para {self.name_model}:")
        logger.info(f"Accuracy Train: {metrics['accuracy_train']:.4f}")
        logger.info(f"Accuracy Test: {metrics['accuracy_test']:.4f}")
        logger.info(f"F1 Score Train: {metrics['f1_train']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"Precision Train: {metrics['precision_train']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall Train: {metrics['recall_train']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        
        return metrics

    def calculate_permutation_importance(self, model, X, y):
        """Calcula la importancia de características usando Permutation Importance"""
        logger.info("Calculando Permutation Importance...")
        try:
            X = pd.DataFrame(X, columns=self.feature_columns)
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
            feature_names = self.feature_columns
            
            logger.info("Permutation Importance calculado exitosamente")
            return importances, feature_names
            
        except Exception as e:
            logger.error(f"Error al calcular Permutation Importance: {str(e)}")
            return None, None

    def get_feature_importances(self, model, vectorizer=None):
        """Extrae importancia de características usando el método más apropiado"""
        logger.info("Extrayendo importancia de características...")
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_, self.feature_columns
            
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_[0]), self.feature_columns
            
        else:
            return self.calculate_permutation_importance(model, self.X_test, self.y_test)

    @abstractmethod
    def train_models(self) -> List[TrainingResponseModel]:
        pass

    def save_model(self, model, filename):
        logger.info(f"Guardando modelo en {filename}")
        joblib.dump(model, filename)
        logger.info("Modelo guardado exitosamente")

# Clase específica para entrenar redes neuronales
class NeuralNetworkTrainer(ModelTrainer):
    def train_models(self) -> List[TrainingResponseModel]:
        logger.info("Iniciando entrenamiento de redes neuronales con análisis de sentimiento")
        results = []

        self.name_model = "Embeddings + Redes Neuronales"
        logger.info(f"Entrenando modelo: {self.name_model}")

        def create_nn_model():
            model = Sequential([
                Dense(64, activation='relu', input_shape=(len(self.feature_columns),)),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model

        nn_model = KerasClassifier(build_fn=create_nn_model, epochs=5, batch_size=32, verbose=1)

        logger.info("Iniciando entrenamiento de la red neuronal...")
        nn_model.fit(self.X_train, self.y_train)
        logger.info("Entrenamiento de red neuronal completado")

        logger.info("Evaluando red neuronal...")
        train_scores = nn_model.score(self.X_train, self.y_train)
        test_scores = nn_model.score(self.X_test, self.y_test)
        y_pred_nn_train = (nn_model.predict(self.X_train) > 0.5).astype('int32')
        y_pred_nn_test = (nn_model.predict(self.X_test) > 0.5).astype('int32')

        metrics = {
            'accuracy_train': train_scores,
            'accuracy_test': test_scores,
            'f1_train': f1_score(self.y_train, y_pred_nn_train, average='weighted'),
            'f1': f1_score(self.y_test, y_pred_nn_test, average='weighted'),
            'precision_train': precision_score(self.y_train, y_pred_nn_train, average='weighted'),
            'precision': precision_score(self.y_test, y_pred_nn_test, average='weighted'),
            'recall_train': recall_score(self.y_train, y_pred_nn_train, average='weighted'),
            'recall': recall_score(self.y_test, y_pred_nn_test, average='weighted')
        }

        feature_importances, feature_names = self.calculate_permutation_importance(nn_model.model, self.X_test, self.y_test)

        logger.info("Guardando modelo de red neuronal...")
        nn_model.model.save('models/nn_embedding_model_sentiment.h5')
        results.append(TrainingResponseModel(
            name_model=self.name_model,
            status="Entrenamiento completado",
            accuracy_train=metrics['accuracy_train'],
            accuracy=metrics['accuracy_test'],
            f1_score=metrics['f1'],
            f1_score_train=metrics['f1_train'],
            precision=metrics['precision'],
            precision_train=metrics['precision_train'],
            recall=metrics['recall'],
            recall_train=metrics['recall_train'],
            message="El modelo se ha entrenado y guardado exitosamente.",
            feature_importances=feature_importances.tolist() if feature_importances is not None else None,
            feature_names=feature_names if feature_names is not None else None
        ))

        logger.info("Entrenamiento de redes neuronales completado exitosamente")
        return results

# Clase para el entrenamiento de modelos con análisis de sentimiento
class SentimentAnalysisModelTrainer(ModelTrainer):
    def train_models(self) -> List[TrainingResponseModel]:
        logger.info("Iniciando entrenamiento de modelos con análisis de sentimiento")
        results = []

        # N-grams + SVM
        self.name_model = "N-grams + SVM"
        logger.info(f"Entrenando modelo: {self.name_model}")
        svm_pipeline = Pipeline([
            ('svm', SVC(probability=True, random_state=RANDOM_SEED))
        ])
        svm_pipeline.fit(self.X_train, self.y_train)
        logger.info(f"Modelo {self.name_model} entrenado exitosamente")
        
        metrics = self.evaluate_model(
            svm_pipeline, 
            self.X_train, self.X_test, 
            self.y_train, self.y_test
        )
        
        feature_importances, feature_names = None, None
        if hasattr(svm_pipeline['svm'], 'coef_'):
            feature_names = self.X.columns.tolist()
            feature_importances = np.abs(svm_pipeline['svm'].coef_[0])
            logger.info(f"Se extrajeron {len(feature_names)} características importantes")

        self.save_model(svm_pipeline, 'models/svm_ngram_model_sentiment.pkl')
        results.append(TrainingResponseModel(
            name_model=self.name_model,
            status="Entrenamiento completado",
            accuracy_train=metrics['accuracy_train'],
            accuracy=metrics['accuracy_test'],
            f1_score=metrics['f1'],
            f1_score_train=metrics['f1_train'],
            precision=metrics['precision'],
            precision_train=metrics['precision_train'],
            recall=metrics['recall'],
            recall_train=metrics['recall_train'],
            message="El modelo se ha entrenado y guardado exitosamente.",
            feature_importances=feature_importances.tolist() if feature_importances is not None else None,
            feature_names=feature_names if feature_names is not None else None,
            y_true=metrics['y_true'].tolist(),
            y_pred=metrics['y_pred'].tolist(),
            y_prob=metrics['y_prob'].tolist() if metrics['y_prob'] is not None else None
        ))

        # N-grams + Regresión Logística
        self.name_model = "N-grams + Regresión Logística"
        logger.info(f"Entrenando modelo: {self.name_model}")
        lr_pipeline = Pipeline([
            ('lr', LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
        ])
        lr_pipeline.fit(self.X_train, self.y_train)
        logger.info(f"Modelo {self.name_model} entrenado exitosamente")
        
        metrics = self.evaluate_model(
            lr_pipeline, 
            self.X_train, self.X_test, 
            self.y_train, self.y_test
        )
        
        feature_importances, feature_names = None, None
        if hasattr(lr_pipeline['lr'], 'coef_'):
            feature_names = self.X.columns.tolist()
            feature_importances = np.abs(lr_pipeline['lr'].coef_[0])
            logger.info(f"Se extrajeron {len(feature_names)} características importantes")

        self.save_model(lr_pipeline, 'models/lr_ngram_model_sentiment.pkl')
        results.append(TrainingResponseModel(
            name_model=self.name_model,
            status="Entrenamiento completado",
            accuracy_train=metrics['accuracy_train'],
            accuracy=metrics['accuracy_test'],
            f1_score=metrics['f1'],
            f1_score_train=metrics['f1_train'],
            precision=metrics['precision'],
            precision_train=metrics['precision_train'],
            recall=metrics['recall'],
            recall_train=metrics['recall_train'],
            message="El modelo se ha entrenado y guardado exitosamente.",
            feature_importances=feature_importances.tolist() if feature_importances is not None else None,
            feature_names=feature_names if feature_names is not None else None,
            y_true=metrics['y_true'].tolist(),
            y_pred=metrics['y_pred'].tolist(),
            y_prob=metrics['y_prob'].tolist() if metrics['y_prob'] is not None else None
        ))

        # Boosting con n-grams de caracteres
        self.name_model = "Boosting (BO) con n-grams de caracteres"
        logger.info(f"Entrenando modelo: {self.name_model}")
        bo_pipeline = Pipeline([
            ('boosting', GradientBoostingClassifier(random_state=RANDOM_SEED))
        ])
        bo_pipeline.fit(self.X_train, self.y_train)
        logger.info(f"Modelo {self.name_model} entrenado exitosamente")
        
        metrics = self.evaluate_model(
            bo_pipeline, 
            self.X_train, self.X_test, 
            self.y_train, self.y_test
        )
        
        feature_importances, feature_names = self.get_feature_importances(
            bo_pipeline['boosting']
        )

        self.save_model(bo_pipeline, 'models/boosting_char_ngram_model_sentiment.pkl')
        results.append(TrainingResponseModel(
            name_model=self.name_model,
            status="Entrenamiento completado",
            accuracy_train=metrics['accuracy_train'],
            accuracy=metrics['accuracy_test'],
            f1_score=metrics['f1'],
            f1_score_train=metrics['f1_train'],
            precision=metrics['precision'],
            precision_train=metrics['precision_train'],
            recall=metrics['recall'],
            recall_train=metrics['recall_train'],
            message="El modelo se ha entrenado y guardado exitosamente.",
            feature_importances=feature_importances.tolist() if feature_importances is not None else None,
            feature_names=feature_names if feature_names is not None else None,
            y_true=metrics['y_true'].tolist(),
            y_pred=metrics['y_pred'].tolist(),
            y_prob=metrics['y_prob'].tolist() if metrics['y_prob'] is not None else None
        ))

        # Embeddings + Redes Neuronales
        nn_trainer = NeuralNetworkTrainer(self.db_path, self.table_name)
        nn_results = nn_trainer.train_models()
        results.extend(nn_results)

        logger.info("Entrenamiento de todos los modelos completado exitosamente")
        return results

# Clase para el entrenamiento de modelos sin análisis de sentimiento
class NoSentimentAnalysisModelTrainer(ModelTrainer):
    def train_models(self) -> List[TrainingResponseModel]:
        logger.info("Iniciando entrenamiento de modelos sin análisis de sentimiento")
        results = []

        # N-grams + SVM
        self.name_model = "N-grams + SVM"
        logger.info(f"Entrenando modelo: {self.name_model}")
        svm_pipeline = Pipeline([
            ('svm', SVC(probability=True, random_state=RANDOM_SEED))
        ])
        svm_pipeline.fit(self.X_train, self.y_train)
        logger.info(f"Modelo {self.name_model} entrenado exitosamente")
        
        metrics = self.evaluate_model(
            svm_pipeline, 
            self.X_train, self.X_test, 
            self.y_train, self.y_test
        )
        
        feature_importances, feature_names = self.get_feature_importances(
            svm_pipeline['svm']
        )

        self.save_model(svm_pipeline, 'models/svm_ngram_model_no_sentiment.pkl')
        results.append(TrainingResponseModel(
            name_model=self.name_model,
            status="Entrenamiento completado",
            accuracy_train=metrics['accuracy_train'],
            accuracy=metrics['accuracy_test'],
            f1_score=metrics['f1'],
            f1_score_train=metrics['f1_train'],
            precision=metrics['precision'],
            precision_train=metrics['precision_train'],
            recall=metrics['recall'],
            recall_train=metrics['recall_train'],
            message="El modelo se ha entrenado y guardado exitosamente.",
            feature_importances=feature_importances.tolist() if feature_importances is not None else None,
            feature_names=feature_names if feature_names is not None else None,
            y_true=metrics['y_true'].tolist(),
            y_pred=metrics['y_pred'].tolist(),
            y_prob=metrics['y_prob'].tolist() if metrics['y_prob'] is not None else None
        ))

        # N-grams + Regresión Logística
        self.name_model = "N-grams + Regresión Logística"
        logger.info(f"Entrenando modelo: {self.name_model}")
        lr_pipeline = Pipeline([
            ('lr', LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
        ])
        lr_pipeline.fit(self.X_train, self.y_train)
        logger.info(f"Modelo {self.name_model} entrenado exitosamente")
        
        metrics = self.evaluate_model(
            lr_pipeline, 
            self.X_train, self.X_test, 
            self.y_train, self.y_test
        )
        
        feature_importances, feature_names = None, None
        if hasattr(lr_pipeline['lr'], 'coef_'):
            feature_names = self.X.columns.tolist()
            feature_importances = np.abs(lr_pipeline['lr'].coef_[0])
            logger.info(f"Se extrajeron {len(feature_names)} características importantes")

        self.save_model(lr_pipeline, 'models/lr_ngram_model_no_sentiment.pkl')
        results.append(TrainingResponseModel(
            name_model=self.name_model,
            status="Entrenamiento completado",
            accuracy_train=metrics['accuracy_train'],
            accuracy=metrics['accuracy_test'],
            f1_score=metrics['f1'],
            f1_score_train=metrics['f1_train'],
            precision=metrics['precision'],
            precision_train=metrics['precision_train'],
            recall=metrics['recall'],
            recall_train=metrics['recall_train'],
            message="El modelo se ha entrenado y guardado exitosamente.",
            feature_importances=feature_importances.tolist() if feature_importances is not None else None,
            feature_names=feature_names if feature_names is not None else None,
            y_true=metrics['y_true'].tolist(),
            y_pred=metrics['y_pred'].tolist(),
            y_prob=metrics['y_prob'].tolist() if metrics['y_prob'] is not None else None
        ))

        # Boosting con n-grams de caracteres
        self.name_model = "Boosting (BO) con n-grams de caracteres"
        logger.info(f"Entrenando modelo: {self.name_model}")
        bo_pipeline = Pipeline([
            ('boosting', GradientBoostingClassifier(random_state=RANDOM_SEED))
        ])
        bo_pipeline.fit(self.X_train, self.y_train)
        logger.info(f"Modelo {self.name_model} entrenado exitosamente")
        
        metrics = self.evaluate_model(
            bo_pipeline, 
            self.X_train, self.X_test, 
            self.y_train, self.y_test
        )
        
        feature_importances, feature_names = self.get_feature_importances(
            bo_pipeline['boosting']
        )

        self.save_model(bo_pipeline, 'models/boosting_char_ngram_model_no_sentiment.pkl')
        results.append(TrainingResponseModel(
            name_model=self.name_model,
            status="Entrenamiento completado",
            accuracy_train=metrics['accuracy_train'],
            accuracy=metrics['accuracy_test'],
            f1_score=metrics['f1'],
            f1_score_train=metrics['f1_train'],
            precision=metrics['precision'],
            precision_train=metrics['precision_train'],
            recall=metrics['recall'],
            recall_train=metrics['recall_train'],
            message="El modelo se ha entrenado y guardado exitosamente.",
            feature_importances=feature_importances.tolist() if feature_importances is not None else None,
            feature_names=feature_names if feature_names is not None else None,
            y_true=metrics['y_true'].tolist(),
            y_pred=metrics['y_pred'].tolist(),
            y_prob=metrics['y_prob'].tolist() if metrics['y_prob'] is not None else None
        ))

        # Embeddings + Redes Neuronales
        nn_trainer = NeuralNetworkTrainer(self.db_path, self.table_name)
        nn_results = nn_trainer.train_models()
        results.extend(nn_results)

        logger.info("Entrenamiento de todos los modelos completado exitosamente")
        return results