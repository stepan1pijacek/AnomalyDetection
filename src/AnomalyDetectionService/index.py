import json
import numpy as np

from flask import Flask, request, jsonify
from sklearn.metrics import mean_squared_error

from src.AnomalyDetectionService.Engine.Engine import Engine
from src.Commons.Data.DataNormalization import DataNormalization as dn
from src.Commons.Models.SpectralAnalysisModel import SpectralAnalysisModel
from src.Commons.Models.SpectralAnomalyModel import SpectralAnomalyModel
from src.AnomalyDetectionService.Data.JSONEncoder_Anomaly import JSONEncoder
from src.AnomalyDetectionEngine.Data.JSONDecoder import JSONDecoder

app = Flask(__name__)

dn = dn()

# Load the trained AutoEncoder model
engine = Engine('/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/Output/AnomalyDetection_weights.best.hdf5')
autoencoder = engine.get_engine()

# Set the status to IDLE initially
status = 'IDLE'


@app.route('/AnomalyDetection', methods=['POST'])
def post_anomaly_detection():
    global status

    # Set the status to PROCESSING
    status = 'PROCESSING'

    spectral_analysis_model = SpectralAnalysisModel()
    anomaly_model = SpectralAnomalyModel()
    empty_array = np.array([])

    # Load the data from the JSON file
    data = request.get_json()
    spectral_analysis_model = json.loads(data, cls=JSONDecoder)

    print(len(spectral_analysis_model.data_y))
    # Normalize the test data
    desired_length = 1016
    data_test_norm = dn.data_normalization(spectral_analysis_model.data_y, desired_length)
    axis_x_norm = dn.data_normalization(spectral_analysis_model.data_x, desired_length)
    axis_x_norm = np.flip(axis_x_norm)

    # Use the trained AutoEncoder to reconstruct the test data
    reconstructed_test_data = autoencoder.predict(data_test_norm)

    # Calculate the Mean Squared Error (MSE) between the original test data and the reconstructions
    mse_per_sample = mean_squared_error(data_test_norm, reconstructed_test_data, multioutput='raw_values')

    # Choose the percentile or standard deviation for the threshold
    threshold_percentile = np.percentile(mse_per_sample, 95)  # Example: 95th percentile

    # Identify anomalies based on the chosen threshold
    anomaly_indices = np.where(mse_per_sample > threshold_percentile)[0]

    anomaly_model.ID = spectral_analysis_model.ID
    anomaly_model.agent = spectral_analysis_model.agent
    anomaly_model.iteration = spectral_analysis_model.iteration
    anomaly_model.concentration = spectral_analysis_model.concentration
    anomaly_model.data_x = axis_x_norm
    anomaly_model.data_y_orig = data_test_norm
    anomaly_model.anomaly_indices = anomaly_indices
    anomaly_model.reconstructed_data = reconstructed_test_data

    status = 'DONE'

    anomaly_json = json.dumps(anomaly_model, cls=JSONEncoder)
    # Return the anomaly object as JSON
    return jsonify(anomaly_json), 200


@app.route('/Status', methods=['GET'])
def get_status():
    global status

    # Return the current status
    return jsonify({'status': status}), 200
