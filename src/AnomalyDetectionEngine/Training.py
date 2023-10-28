import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from pickle import dump
from Utilities.FileLoader import FileLoader as fl
from AnomalyDetectors.AnomalyDetectionModel import AnomalyDetectionAutoEncoder
from Data.DataAugmentation import DataAugmentaion as da
from src.Commons.Data.DataProcessing import DataProcessing as dp
from src.Commons.Settings.GlobalSettings import GlobalSettings
from tensorflow.python.keras.utils.version_utils import callbacks


def main():
    training = True
    file_loader = fl()
    data_processing = dp()

    data_augmentation = da(noise_level=GlobalSettings.NOISE_LEVEL,
                           drift_factor=GlobalSettings.DRIFT_FACTOR,
                           scale_factor=GlobalSettings.SCALE_FACTOR,
                           num_augmented_samples=GlobalSettings.NUM_AUGMENTED_SAMPLES)

    if (GlobalSettings.CLEAN_DATA):
        # Clean anomalous data
        data_processing.data_raw_to_clean_from_files(
            output_folder='/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/CleanData',
            data_file_list=GlobalSettings.DATA_FILES_LIST_ANOMALOUS)

        # Clean control data
        data_processing.data_raw_to_clean_from_files(
            output_folder='/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/CleanData/kontrol',
            data_file_list=GlobalSettings.DATA_FILES_LIST_CONTROL)

    # Assuming data_loader() returns train_data and test_data as lists of arrays
    train_data, test_data, axis_x = file_loader.json_data_loader()

    augmented_train_data = data_augmentation.apply_data_augmentaion(train_data)
    # Pad or truncate the arrays to the same length
    max_length = max(len(arr) for arr in train_data + test_data)
    max_length = max_length + 1

    train_data = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in train_data]
    augmented_train_data = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in augmented_train_data]

    test_data = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in test_data]
    axis_x = [np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in axis_x]

    # TODO: Move to the globall settings, and the stuff into its own class this is a mess
    aug_control_combined = (train_data + augmented_train_data)
    aug_control_combined = np.array(aug_control_combined, dtype=np.float32)
    total_samples = aug_control_combined.shape[0]

    num_validation_samples = int(total_samples * GlobalSettings.VALIDATION_PERCENTAGE)

    train_data_20_missing = aug_control_combined[:-num_validation_samples]
    validation_data = aug_control_combined[-num_validation_samples:]

    # Convert the lists of arrays into NumPy arrays and normalize the data for better training
    train_data_array = np.array(train_data_20_missing, dtype=np.float32)

    validation_data_array = np.array(validation_data, dtype=np.float32)

    test_data_array = np.array(test_data, dtype=np.float32)

    # Initialize the AutoEncoder model
    input_size = 1016  # The size of each input array

    autoencoder = AnomalyDetectionAutoEncoder.create_autoencoder(GlobalSettings.INPUT_SIZE,
                                                                 GlobalSettings.CODE_SIZE)

    # Choose a loss function and an optimizer
    loss_function = tf.keras.losses.MeanSquaredError()  # Mean Squared Error loss function
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=GlobalSettings.LEARNING_RATE,
                                            momentum=0.3)

    if (os.path.isfile(GlobalSettings.WEIGHT_PATH)):
        autoencoder.load_weights(GlobalSettings.WEIGHT_PATH)

    autoencoder.compile(optimizer=optimizer,
                        loss=loss_function,
                        metrics=[keras.metrics.RootMeanSquaredError(name='RMSE'),
                                 keras.metrics.AUC(name='auc'),
                                 keras.metrics.mean_squared_error])
    autoencoder.build(input_shape=input_size)

    if (GlobalSettings.SHOW_ENCODER_LAYERS):
        # Extract the encoder and decoder models
        encoder_model = autoencoder.layers[0]
        decoder_model = autoencoder.layers[1]

        # Get summaries for encoder and decoder
        encoder_model.summary()
        decoder_model.summary()

    if (training == True):
        log = callbacks.CSVLogger('/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/Output/log.csv')
        checkpoint = callbacks.ModelCheckpoint(GlobalSettings.WEIGHT_PATH, monitor='auc', mode='max',
                                               save_best_only=True, save_weights_only=True, verbose=1)
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * (0.0001 ** epoch))

        history = autoencoder.fit(train_data_array,
                                  train_data_array,
                                  batch_size=GlobalSettings.BATCH_SIZE,
                                  epochs=GlobalSettings.NUM_EPOCHS,
                                  verbose=1,
                                  shuffle=False,
                                  steps_per_epoch=GlobalSettings.STEPS_PER_EPOCH,
                                  validation_data=(validation_data_array, validation_data_array),
                                  callbacks=[log, checkpoint, lr_decay])

        loss = autoencoder.evaluate(validation_data_array,
                                    validation_data_array,
                                    steps=GlobalSettings.VALIDATION_STEPS)

        print(loss)

        with open(f'/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/Output/history.txt',
                  'wb') as handle:
            dump(history.history, handle)

        autoencoder.save_weights(GlobalSettings.WEIGHT_PATH)
        # Saving the model for future use
        autoencoder.save("autoencoder_model")

    return 0


if __name__ == "__main__":
    main()
