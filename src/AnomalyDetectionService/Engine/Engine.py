import tensorflow as tf

from tensorflow import keras
from src.AnomalyDetectionEngine.AnomalyDetectors.AnomalyDetectionModel import AnomalyDetectionAutoEncoder
from src.Commons.Settings.GlobalSettings import GlobalSettings


class Engine:
    def __init__(self, weight_path):
        self.weight_path = weight_path 

    def get_engine(self):
        # Load the trained AutoEncoder model
        autoencoder = AnomalyDetectionAutoEncoder.create_autoencoder(GlobalSettings.INPUT_SIZE, GlobalSettings.CODE_SIZE)
        autoencoder.load_weights(self.weight_path)
        autoencoder.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=GlobalSettings.LEARNING_RATE, momentum=0.3), 
                            loss=tf.keras.losses.MeanSquaredError(), 
                            metrics=[keras.metrics.RootMeanSquaredError(name='RMSE'), keras.metrics.AUC(name='auc'), keras.metrics.mean_squared_error])
        autoencoder.build(input_shape=1016)
        return autoencoder
