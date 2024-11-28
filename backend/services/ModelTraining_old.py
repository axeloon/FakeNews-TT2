import pandas as pd
import numpy as np
import logging
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from abc import ABC, abstractmethod
from typing import List

from backend.database.connection import SessionLocal
from backend.utils.response_models import TrainingResponseModel

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


# Clase abstracta base para el entrenamiento de modelos
class ModelTrainer(ABC):
    def __init__(self, db_path, table_name):
        logger.info(f"Inicializando ModelTrainer con db_path: {db_path} y table_name: {table_name}")
        self.db_path = db_path
        self.table_name = table_name
        self.data = self.load_data()
        logger.info(f"Datos cargados exitosamente. Tamaño del dataset: {len(self.data)}")
        
        self.X = self.data['content']
        self.y = self.data['is_false']
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        logger.info(f"División de datos completada. Conjunto de entrenamiento: {len(self.X_train)}, Conjunto de prueba: {len(self.X_test)}")
        
        self.name_model = None
        self.metrics = {}

    def load_data(self):
        logger.info("Cargando datos desde la base de datos...")
        # Utilizar la conexión existente para cargar los datos desde la base de datos
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
        
        metrics = {
            'accuracy_train': accuracy_score(y_train, y_pred_train),
            'accuracy_test': accuracy_score(y_test, y_pred_test),
            'f1': f1_score(y_test, y_pred_test, average='weighted'),
            'precision': precision_score(y_test, y_pred_test, average='weighted'),
            'recall': recall_score(y_test, y_pred_test, average='weighted')
        }
        
        logger.info(f"Métricas de evaluación para {self.name_model}:")
        logger.info(f"Accuracy Train: {metrics['accuracy_train']:.4f}")
        logger.info(f"Accuracy Test: {metrics['accuracy_test']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        
        return metrics

    def get_feature_importances(self, model, vectorizer):
        """Extrae importancia de características si está disponible"""
        logger.info("Extrayendo importancia de características...")
        if hasattr(model, 'feature_importances_'):
            feature_names = vectorizer.get_feature_names_out()
            logger.info(f"Se encontraron {len(feature_names)} características importantes")
            return model.feature_importances_, feature_names
        logger.info("No se encontraron características importantes para este modelo")
        return None, None

    @abstractmethod
    def train_models(self) -> List[TrainingResponseModel]:
        pass

    def save_model(self, model, filename):
        # Guardar el modelo entrenado en un archivo
        logger.info(f"Guardando modelo en {filename}")
        joblib.dump(model, filename)
        logger.info("Modelo guardado exitosamente")

# Clase para el entrenamiento de modelos con análisis de sentimiento
class SentimentAnalysisModelTrainer(ModelTrainer):
    def train_models(self) -> List[TrainingResponseModel]:
        logger.info("Iniciando entrenamiento de modelos con análisis de sentimiento")
        results = []

        # N-grams + SVM
        self.name_model = "N-grams + SVM"
        logger.info(f"Entrenando modelo: {self.name_model}")
        svm_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),  # Vectorizar el texto usando n-grams
            ('svm', SVC())  # Clasificador SVM
        ])
        svm_pipeline.fit(self.X_train, self.y_train)  # Entrenar el modelo
        logger.info(f"Modelo {self.name_model} entrenado exitosamente")
        
        # Evaluar modelo
        metrics = self.evaluate_model(
            svm_pipeline, 
            self.X_train, self.X_test, 
            self.y_train, self.y_test
        )
        
        self.save_model(svm_pipeline, 'models/svm_ngram_model_sentiment.pkl')  # Guardar el modelo
        results.append(TrainingResponseModel(
            name_model=self.name_model,
            status="Entrenamiento completado",
            accuracy_train=metrics['accuracy_train'],
            accuracy=metrics['accuracy_test'],
            f1_score=metrics['f1'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            message="El modelo se ha entrenado y guardado exitosamente.",
            feature_importances=None,  # SVM no proporciona importancia de características
            feature_names=None
        ))

        # N-grams + Regresión Logística
        self.name_model = "N-grams + Regresión Logística"
        logger.info(f"Entrenando modelo: {self.name_model}")
        lr_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),  # Vectorizar el texto usando n-grams
            ('lr', LogisticRegression())  # Clasificador de Regresión Logística
        ])
        lr_pipeline.fit(self.X_train, self.y_train)  # Entrenar el modelo
        logger.info(f"Modelo {self.name_model} entrenado exitosamente")
        
        metrics = self.evaluate_model(
            lr_pipeline, 
            self.X_train, self.X_test, 
            self.y_train, self.y_test
        )
        
        # Obtener coeficientes como importancia de características
        feature_importances, feature_names = None, None
        if hasattr(lr_pipeline['lr'], 'coef_'):
            feature_names = lr_pipeline['tfidf'].get_feature_names_out()
            feature_importances = np.abs(lr_pipeline['lr'].coef_[0])
            logger.info(f"Se extrajeron {len(feature_names)} características importantes")

        self.save_model(lr_pipeline, 'models/lr_ngram_model_sentiment.pkl')  # Guardar el modelo
        results.append(TrainingResponseModel(
            name_model=self.name_model,
            status="Entrenamiento completado",
            accuracy_train=metrics['accuracy_train'],
            accuracy=metrics['accuracy_test'],
            f1_score=metrics['f1'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            message="El modelo se ha entrenado y guardado exitosamente.",
            feature_importances=feature_importances.tolist() if feature_importances is not None else None,
            feature_names=feature_names.tolist() if feature_names is not None else None
        ))

        # Boosting con n-grams de caracteres
        self.name_model = "Boosting (BO) con n-grams de caracteres"
        logger.info(f"Entrenando modelo: {self.name_model}")
        bo_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 4))),  # Vectorizar el texto usando n-grams de caracteres
            ('boosting', GradientBoostingClassifier())  # Clasificador Boosting
        ])
        bo_pipeline.fit(self.X_train, self.y_train)  # Entrenar el modelo
        logger.info(f"Modelo {self.name_model} entrenado exitosamente")
        
        metrics = self.evaluate_model(
            bo_pipeline, 
            self.X_train, self.X_test, 
            self.y_train, self.y_test
        )
        
        # Obtener importancia de características
        feature_importances, feature_names = self.get_feature_importances(
            bo_pipeline['boosting'],
            bo_pipeline['tfidf']
        )

        self.save_model(bo_pipeline, 'models/boosting_char_ngram_model_sentiment.pkl')  # Guardar el modelo
        results.append(TrainingResponseModel(
            name_model=self.name_model,
            status="Entrenamiento completado",
            accuracy_train=metrics['accuracy_train'],
            accuracy=metrics['accuracy_test'],
            f1_score=metrics['f1'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            message="El modelo se ha entrenado y guardado exitosamente.",
            feature_importances=feature_importances.tolist() if feature_importances is not None else None,
            feature_names=feature_names.tolist() if feature_names is not None else None
        ))

        # Embeddings + Redes Neuronales
        self.name_model = "Embeddings + Redes Neuronales"
        logger.info(f"Entrenando modelo: {self.name_model}")
        logger.info("Preparando datos para la red neuronal...")
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(self.X_train)
        X_train_seq = tokenizer.texts_to_sequences(self.X_train)
        X_test_seq = tokenizer.texts_to_sequences(self.X_test)
        X_train_pad = pad_sequences(X_train_seq, maxlen=100)
        X_test_pad = pad_sequences(X_test_seq, maxlen=100)
        logger.info("Datos preparados exitosamente")

        logger.info("Construyendo modelo de red neuronal...")
        nn_model = Sequential()
        nn_model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))
        nn_model.add(LSTM(64))
        nn_model.add(Dense(1, activation='sigmoid'))

        nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logger.info("Iniciando entrenamiento de la red neuronal...")
        history = nn_model.fit(
            X_train_pad, 
            self.y_train, 
            epochs=5, 
            batch_size=32, 
            validation_split=0.1,
            verbose=1
        )
        logger.info("Entrenamiento de red neuronal completado")

        # Evaluar en conjunto de entrenamiento y prueba
        logger.info("Evaluando red neuronal...")
        train_scores = nn_model.evaluate(X_train_pad, self.y_train, verbose=0)
        test_scores = nn_model.evaluate(X_test_pad, self.y_test, verbose=0)
        y_pred_nn = (nn_model.predict(X_test_pad) > 0.5).astype('int32')

        metrics = {
            'accuracy_train': train_scores[1],
            'accuracy_test': test_scores[1],
            'f1': f1_score(self.y_test, y_pred_nn, average='weighted'),
            'precision': precision_score(self.y_test, y_pred_nn, average='weighted'),
            'recall': recall_score(self.y_test, y_pred_nn, average='weighted')
        }

        logger.info("Guardando modelo de red neuronal...")
        nn_model.save('models/nn_embedding_model_sentiment.h5')
        results.append(TrainingResponseModel(
            name_model=self.name_model,
            status="Entrenamiento completado",
            accuracy_train=metrics['accuracy_train'],
            accuracy=metrics['accuracy_test'],
            f1_score=metrics['f1'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            message="El modelo se ha entrenado y guardado exitosamente.",
            feature_importances=None,  # Las redes neuronales no proporcionan importancia de características directamente
            feature_names=None
        ))

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
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('svm', SVC())
        ])
        svm_pipeline.fit(self.X_train, self.y_train)
        logger.info(f"Modelo {self.name_model} entrenado exitosamente")
        
        metrics = self.evaluate_model(
            svm_pipeline, 
            self.X_train, self.X_test, 
            self.y_train, self.y_test
        )
        
        self.save_model(svm_pipeline, 'models/svm_ngram_model_no_sentiment.pkl')
        results.append(TrainingResponseModel(
            name_model=self.name_model,
            status="Entrenamiento completado",
            accuracy_train=metrics['accuracy_train'],
            accuracy=metrics['accuracy_test'],
            f1_score=metrics['f1'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            message="El modelo se ha entrenado y guardado exitosamente.",
            feature_importances=None,
            feature_names=None
        ))

        # N-grams + Regresión Logística
        self.name_model = "N-grams + Regresión Logística"
        logger.info(f"Entrenando modelo: {self.name_model}")
        lr_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('lr', LogisticRegression())
        ])
        lr_pipeline.fit(self.X_train, self.y_train)
        logger.info(f"Modelo {self.name_model} entrenado exitosamente")
        
        metrics = self.evaluate_model(
            lr_pipeline, 
            self.X_train, self.X_test, 
            self.y_train, self.y_test
        )
        
        # Obtener coeficientes como importancia de características
        feature_importances, feature_names = None, None
        if hasattr(lr_pipeline['lr'], 'coef_'):
            feature_names = lr_pipeline['tfidf'].get_feature_names_out()
            feature_importances = np.abs(lr_pipeline['lr'].coef_[0])
            logger.info(f"Se extrajeron {len(feature_names)} características importantes")

        self.save_model(lr_pipeline, 'models/lr_ngram_model_no_sentiment.pkl')
        results.append(TrainingResponseModel(
            name_model=self.name_model,
            status="Entrenamiento completado",
            accuracy_train=metrics['accuracy_train'],
            accuracy=metrics['accuracy_test'],
            f1_score=metrics['f1'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            message="El modelo se ha entrenado y guardado exitosamente.",
            feature_importances=feature_importances.tolist() if feature_importances is not None else None,
            feature_names=feature_names.tolist() if feature_names is not None else None
        ))

        # Boosting con n-grams de caracteres
        self.name_model = "Boosting (BO) con n-grams de caracteres"
        logger.info(f"Entrenando modelo: {self.name_model}")
        bo_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 4))),
            ('boosting', GradientBoostingClassifier())
        ])
        bo_pipeline.fit(self.X_train, self.y_train)
        logger.info(f"Modelo {self.name_model} entrenado exitosamente")
        
        metrics = self.evaluate_model(
            bo_pipeline, 
            self.X_train, self.X_test, 
            self.y_train, self.y_test
        )
        
        # Obtener importancia de características
        feature_importances, feature_names = self.get_feature_importances(
            bo_pipeline['boosting'],
            bo_pipeline['tfidf']
        )

        self.save_model(bo_pipeline, 'models/boosting_char_ngram_model_no_sentiment.pkl')
        results.append(TrainingResponseModel(
            name_model=self.name_model,
            status="Entrenamiento completado",
            accuracy_train=metrics['accuracy_train'],
            accuracy=metrics['accuracy_test'],
            f1_score=metrics['f1'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            message="El modelo se ha entrenado y guardado exitosamente.",
            feature_importances=feature_importances.tolist() if feature_importances is not None else None,
            feature_names=feature_names.tolist() if feature_names is not None else None
        ))

        # Embeddings + Redes Neuronales
        self.name_model = "Embeddings + Redes Neuronales"
        logger.info(f"Entrenando modelo: {self.name_model}")
        logger.info("Preparando datos para la red neuronal...")
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(self.X_train)
        X_train_seq = tokenizer.texts_to_sequences(self.X_train)
        X_test_seq = tokenizer.texts_to_sequences(self.X_test)
        X_train_pad = pad_sequences(X_train_seq, maxlen=100)
        X_test_pad = pad_sequences(X_test_seq, maxlen=100)
        logger.info("Datos preparados exitosamente")

        logger.info("Construyendo modelo de red neuronal...")
        nn_model = Sequential()
        nn_model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))
        nn_model.add(LSTM(64))
        nn_model.add(Dense(1, activation='sigmoid'))

        nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logger.info("Iniciando entrenamiento de la red neuronal...")
        history = nn_model.fit(
            X_train_pad, 
            self.y_train, 
            epochs=5, 
            batch_size=32, 
            validation_split=0.1,
            verbose=1
        )
        logger.info("Entrenamiento de red neuronal completado")

        train_scores = nn_model.evaluate(X_train_pad, self.y_train, verbose=0)
        test_scores = nn_model.evaluate(X_test_pad, self.y_test, verbose=0)
        y_pred_nn = (nn_model.predict(X_test_pad) > 0.5).astype('int32')

        metrics = {
            'accuracy_train': train_scores[1],
            'accuracy_test': test_scores[1],
            'f1': f1_score(self.y_test, y_pred_nn, average='weighted'),
            'precision': precision_score(self.y_test, y_pred_nn, average='weighted'),
            'recall': recall_score(self.y_test, y_pred_nn, average='weighted')
        }

        logger.info("Guardando modelo de red neuronal...")
        nn_model.save('models/nn_embedding_model_no_sentiment.h5')
        results.append(TrainingResponseModel(
            name_model=self.name_model,
            status="Entrenamiento completado",
            accuracy_train=metrics['accuracy_train'],
            accuracy=metrics['accuracy_test'],
            f1_score=metrics['f1'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            message="El modelo se ha entrenado y guardado exitosamente.",
            feature_importances=None,
            feature_names=None
        ))

        logger.info("Entrenamiento de todos los modelos completado exitosamente")
        return results