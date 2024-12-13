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
        """Proceso de fine-tuning específico para redes neuronales"""
        try:
            model = self.create_model()
            
            # Calcular pesos de clase para manejar desbalance
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(self.y_train),
                y=self.y_train
            )
            class_weight_dict = dict(enumerate(class_weights))
            
            # Callbacks para mejorar el entrenamiento
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    min_delta=0.001
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001
                )
            ]
            
            # Entrenar el modelo
            history = model.fit(
                self.X_train, self.y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1
            )
            
            # Crear un wrapper personalizado que mantenga el modelo original
            class CustomKerasClassifier(KerasClassifier):
                def predict(self, X):
                    predictions = self.model.predict(X)
                    return (predictions > 0.5).astype(int).ravel()
                    
                def predict_proba(self, X):
                    return self.model.predict(X).ravel()
            
            # Envolver el modelo entrenado
            wrapped_model = CustomKerasClassifier(build_fn=lambda: model)
            wrapped_model.model = model
            wrapped_model.history = history.history
            wrapped_model.classes_ = np.unique(self.y_train)
            
            # Verificar que las predicciones funcionan
            try:
                test_pred = wrapped_model.predict(self.X_test[:1])
                test_prob = wrapped_model.predict_proba(self.X_test[:1])
                logger.info(f"Test prediction shape: {test_pred.shape}")
                logger.info(f"Test probability shape: {test_prob.shape}")
            except Exception as e:
                logger.error(f"Error en predicción de prueba: {str(e)}")
            
            return wrapped_model
            
        except Exception as e:
            logger.error(f"Error en fine-tuning de red neuronal: {str(e)}")
            raise