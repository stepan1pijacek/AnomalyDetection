import tensorflow as tf


class AnomalyDetectionAutoEncoder():
    @staticmethod
    # Define the AutoEncoder architecture using TensorFlow's Keras API
    def create_autoencoder(input_size, code_size):
        encoder = tf.keras.Sequential([
            tf.keras.layers.Reshape((input_size, 1), input_shape=(input_size,)),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.MaxPooling1D(pool_size=2, padding='valid'),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(pool_size=2, padding='valid'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(pool_size=2, padding='valid'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(input_size, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(code_size, activation='relu')
        ])

        decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(code_size, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(input_size, activation='relu'),
            tf.keras.layers.Dense((input_size // (2**3)) * 128, activation='relu'),
            tf.keras.layers.Reshape((input_size // (2**3), 128)),
            tf.keras.layers.UpSampling1D(size=2),
            tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=3, activation='relu', padding='same'),
            tf.keras.layers.UpSampling1D(size=2),
            tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=3, activation='relu', padding='same'),
            tf.keras.layers.UpSampling1D(size=2),
            tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=3, activation='relu', padding='same'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))
        ])

        autoencoder = tf.keras.Sequential([encoder, decoder])
        return autoencoder
