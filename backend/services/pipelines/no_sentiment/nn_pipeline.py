import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils.class_weight import compute_class_weight
from logging import getLogger
from .base_pipeline import NoSentimentBasePipeline
from backend.constant import RANDOM_SEED

logger = getLogger(__name__)

class NeuralNetworkNoSentimentPipeline(NoSentimentBasePipeline):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.model = None
        self.best_model = None

    def get_param_grid(self):
        return None
    
    def create_model(self):
        tf.random.set_seed(RANDOM_SEED)
        
        model = Sequential([
            # Capa de entrada con normalización
            Dense(128, input_shape=(self.X_train.shape[1],)),
            BatchNormalization(),
            
            # Primera capa oculta
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Segunda capa oculta
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Capa de salida
            Dense(1, activation='sigmoid')
        ])
        
        optimizer = Adam(
            learning_rate=0.001,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fine_tune(self):
        model = self.create_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=16,
            callbacks=callbacks,
            class_weight=class_weight_dict
        )
        
        self.model = model
        self.best_model = model
        return self

    def predict(self, X):
        """
        Realiza predicciones binarias
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        """
        Realiza predicciones de probabilidad
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        return self.model.predict(X).flatten()