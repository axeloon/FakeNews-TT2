from .base_pipeline import BaseModelPipeline
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class NeuralNetworkPipeline(BaseModelPipeline):
    def get_param_grid(self):
        # Las redes neuronales usan un enfoque diferente
        return None
    
    def create_model(self):
        inputs = tf.keras.Input(shape=(self.X_train.shape[1],))
        
        # Normalización inicial
        x = tf.keras.layers.BatchNormalization()(inputs)
        
        # Rama principal
        main = tf.keras.layers.Dense(128, activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.01))(x)
        main = tf.keras.layers.BatchNormalization()(main)
        main = tf.keras.layers.Dropout(0.5)(main)
        
        # Rama residual 1
        res1 = tf.keras.layers.Dense(64, activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.01))(main)
        res1 = tf.keras.layers.BatchNormalization()(res1)
        res1 = tf.keras.layers.Dropout(0.3)(res1)
        
        # Rama residual 2 (skip connection)
        res2 = tf.keras.layers.Dense(32, activation='relu')(main)
        res2 = tf.keras.layers.BatchNormalization()(res2)
        
        # Fusión de ramas
        concat = tf.keras.layers.Concatenate()([
            tf.keras.layers.Dense(16)(res1),
            tf.keras.layers.Dense(16)(res2)
        ])
        
        # Capa de atención
        attention = tf.keras.layers.Dense(32, activation='tanh')(concat)
        attention = tf.keras.layers.Dense(1, activation='sigmoid')(attention)
        attended = tf.keras.layers.Multiply()([concat, attention])
        
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(attended)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0005,  # Learning rate más bajo
            clipnorm=0.5,  # Clipping más agresivo
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
                patience=10,  # Más paciencia
                restore_best_weights=True,
                min_delta=0.001  # Más sensible a mejoras pequeñas
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        # Ajuste de pesos de clase más agresivo
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        class_weight_dict = dict(enumerate(class_weights * 1.2))  # Aumentar el efecto
        
        model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=100,  # Más épocas
            batch_size=16,  # Batch size más pequeño
            callbacks=callbacks,
            class_weight=class_weight_dict
        )
        
        self.best_model = model
        return model 