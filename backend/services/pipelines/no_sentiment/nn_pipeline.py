from .base_pipeline import NoSentimentBasePipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from backend.constant import RANDOM_SEED

class NeuralNetworkNoSentimentPipeline(NoSentimentBasePipeline):
    def get_param_grid(self):
        return None
    
    def create_model(self):
        tf.random.set_seed(RANDOM_SEED)
        
        def create_nn_model():
            model = Sequential([
                Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return model
        
        return KerasClassifier(build_fn=create_nn_model, epochs=5, batch_size=32, verbose=1) 