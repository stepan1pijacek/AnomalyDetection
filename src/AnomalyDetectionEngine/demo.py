import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.metrics import mean_squared_error
from AnomalyDetectors.AnomalyDetectionModel import AnomalyDetectionAutoEncoder
from src.Commons.Settings.GlobalSettings import GlobalSettings
from Utilities.FileLoader import FileLoader as fl


#bs = keras.models.load_model('/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/autoencoder_model')
#bs.summary()

autoencoder = AnomalyDetectionAutoEncoder.create_autoencoder(GlobalSettings.INPUT_SIZE,
         GlobalSettings.CODE_SIZE)

    # Choose a loss function and an optimizer
loss_function = tf.keras.losses.MeanSquaredError()  # Mean Squared Error loss function
optimizer = tf.keras.optimizers.RMSprop(learning_rate=GlobalSettings.LEARNING_RATE,
         momentum=0.3)

autoencoder.load_weights('/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/Output/AnomalyDetection_weights.best.hdf5')

autoencoder.compile(optimizer=optimizer, 
        loss=loss_function,
        metrics=[keras.metrics.RootMeanSquaredError(name='RMSE'),
             keras.metrics.AUC(name='auc'),
             keras.metrics.mean_squared_error])
autoencoder.build(input_shape=1016)

file_loader = fl()
# Use the trained AutoEncoder to reconstruct the test data
data_test, axis_x = file_loader.get_data_json('/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/CleanData/FS232_VOR 0.25_50x_15sa.json')
#data_comp, axis_x = file_loader.get_data_json('/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/CleanData/CCM8261_AMP 0.2_50x_15sh.json')
    # Desired length after padding
desired_length = 1016

    # Calculate the amount of padding needed on each side
padding = desired_length - len(data_test)

    #Normilize data
data_test_norm = [np.pad(data_test, (0, 1016 - len(data_test)), mode='constant')]
axis_x_norm = [[np.pad(axis_x, (0, 1016 - len(axis_x)), mode='constant')]]
data_test_norm = np.array(data_test_norm, dtype=np.float32)
axis_x_norm = np.array(axis_x_norm, dtype=np.float32)
axis_x_norm = np.flip(axis_x_norm)

# Predict using the autoencoder
reconstructed_test_data = autoencoder.predict(data_test_norm)

# Calculate the Mean Squared Error (MSE) between the original test data and the reconstructions
# Calculate the Mean Squared Error (MSE) between the original test data and the reconstructions
mse_per_sample = mean_squared_error(data_test_norm, reconstructed_test_data, multioutput='raw_values')

# Analyze the MSE distribution
plt.hist(mse_per_sample, bins=50, density=True, alpha=0.6, color='blue')
plt.xlabel('MSE Value')
plt.ylabel('Density')
plt.title('MSE Distribution')

# Choose the percentile or standard deviation for the threshold
threshold_percentile = np.percentile(mse_per_sample, 95)  # Example: 95th percentile
# Or use standard deviation: threshold_std = np.mean(mse_per_sample) + 2 * np.std(mse_per_sample)

# Set the threshold
plt.axvline(threshold_percentile, color='red', linestyle='dashed', linewidth=2, label=f'Threshold (95th Percentile)')
plt.legend()

# Identify anomalies based on the chosen threshold
anomaly_indices = np.where(mse_per_sample > threshold_percentile)[0]

# Plot the original data and detected anomalies
plt.figure(figsize=(10, 6))

# Plot the original data
for i in range(data_test_norm.shape[0]):
    plt.plot(np.reshape(axis_x_norm, [1, 1016])[i], data_test_norm[i], label=f'Spectrum {i + 1}', alpha=0.5, color='blue')

bs = data_test_norm.shape[1]
random_arr = []
random_arr_two = []

if len(anomaly_indices) > 0:
    plt.scatter([np.reshape(axis_x_norm, [1, 1016])[0][idx] for idx in anomaly_indices], 
                [data_test_norm[0][idx] for idx in anomaly_indices],
                marker='+', s=50, c="red", label='anomaly')

plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.title('Anomaly Detection in Raman Spectroscopy Data')
plt.legend()
plt.grid(True)

plt.show()