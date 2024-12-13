from .base_pipeline import SentimentBasePipeline
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from backend.constant import RANDOM_SEED

class NeuralNetworkSentimentPipeline(SentimentBasePipeline):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.model = None
        self.best_model = None

    def get_param_grid(self):
        return None
    
    def create_model(self):
        tf.random.set_seed(RANDOM_SEED)
        
        inputs = tf.keras.Input(shape=(self.X_train.shape[1],))
        
        x = tf.keras.layers.BatchNormalization()(inputs)
        
        main = tf.keras.layers.Dense(128, activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.01))(x)
        main = tf.keras.layers.BatchNormalization()(main)
        main = tf.keras.layers.Dropout(0.5)(main)
        
        res1 = tf.keras.layers.Dense(64, activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.01))(main)
        res1 = tf.keras.layers.BatchNormalization()(res1)
        res1 = tf.keras.layers.Dropout(0.3)(res1)
        
        res2 = tf.keras.layers.Dense(32, activation='relu')(main)
        res2 = tf.keras.layers.BatchNormalization()(res2)
        
        concat = tf.keras.layers.Concatenate()([
            tf.keras.layers.Dense(16)(res1),
            tf.keras.layers.Dense(16)(res2)
        ])
        
        attention = tf.keras.layers.Dense(32, activation='tanh')(concat)
        attention = tf.keras.layers.Dense(1, activation='sigmoid')(attention)
        attended = tf.keras.layers.Multiply()([concat, attention])
        
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(attended)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0005,
            clipnorm=0.5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
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
        class_weight_dict = dict(enumerate(class_weights * 1.2))
        
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