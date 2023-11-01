import json

import matplotlib.pyplot as plt
import numpy as np
import requests

from src.Commons.Data.DataParser import DataParser as dp
from src.AnomalyDetectionEngine.Data.JSONEncoder import JSONEncoder
from src.AnomalyDetectionService.Data.JSONDecoder_Anomaly import JSONDecoder
from src.Commons.Models.SpectralAnomalyModel import SpectralAnomalyModel
from src.Commons.Data.DataProcessing import DataProcessing as rc
from src.Commons.Data.DataNormalization import DataNormalization as dn


def process_files(file_paths, folder_path):
    data_parser = dp(file_paths, folder_path)

    return data_parser.to_data_model()


def api_call_anomaly_detection(data_to_process):
    json_string = json.dumps(data_to_process, cls=JSONEncoder)

    output = requests.post('http://127.0.0.1:5000/AnomalyDetection', json=json_string)
    output_formatted = json.loads(output.text, cls=JSONDecoder)

    return output_formatted


def process_data(data_list):
    data_processing = rc()
    return data_processing.data_raw_to_clean_from_models(model_list=data_list)


def normalize_data(data_list, desired_length):
    data_normalization = dn()
    process_data_list = [SpectralAnomalyModel()]
    for data in data_list:
        data.data_x = data_normalization.data_normalization(data=data.data_x, desired_length=desired_length)
        data.data_y = data_normalization.data_normalization(data=data.data_y, desired_length=desired_length)

        process_data_list.append(data)

    return process_data_list


def main():
    processed_anomalies = [SpectralAnomalyModel()]
    control_anomalies = SpectralAnomalyModel()

    # Process anomalous data
    # C:\Users\stepan.pijacek\Documents\GitHub\AnomalyDetection\DemoTestData\CCM8261_AMP 0.5_50x_15s.txt
    anomalous_data = process_files(['CCM8261_FLU 1_50x_15sg.txt'],
                                   'C:\\Users\\stepan.pijacek\\Documents\\GitHub\\AnomalyDetection\\DemoTestData')

    # Process control data
    control_data = process_files(['CCM8261_FLU 1_50x_15sg.txt'],
                                 'C:\\Users\\stepan.pijacek\\Documents\\GitHub\\AnomalyDetection\\DemoTestData')

    # Clear up the data
    anomalous_data = process_data(anomalous_data)
    control_data = process_data(control_data)

    for anomalies in anomalous_data:
        model = SpectralAnomalyModel()
        json_string_processed = api_call_anomaly_detection(anomalies)

        model = json.loads(json_string_processed, cls=JSONDecoder)
        processed_anomalies.append(model)

    processed_anomalies[0].data_x[0] = np.reshape(processed_anomalies[0].data_x[0], [1, 1016])

    print(processed_anomalies)
    # Show control data
    plt.figure(figsize=(30, 20))
    plt.subplot(4, 1, 1)
    plt.plot(processed_anomalies[0].data_y_orig[0], label='original')
    plt.title('Original Data')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(4, 1, 2)
    plt.plot(processed_anomalies[0].reconstructed_data[0], label="Reconstructed data from AnomalyDetection Engine")
    plt.title("Reconstructed data")

    difference = np.subtract(processed_anomalies[0].data_y_orig[0], processed_anomalies[0].reconstructed_data[0])
    plt.subplot(4, 1, 3)
    plt.plot(difference, label="Difference between original and reconstructed data")
    plt.title("Difference")
    plt.legend()

    for anomaly in processed_anomalies:
        # Plot the original data
        for i in range(1):
            plt.subplot(4, 1, 4)
            plt.plot(anomaly.data_x[0],
                     anomaly.data_y_orig[i],
                     label=f'Spectrum {anomaly.ID, anomaly.agent, anomaly.concentration, anomaly.iteration}', alpha=0.5,
                     color='blue')

        if len(anomaly.anomaly_indices) > 0:
            plt.subplot(4, 1, 4)
            plt.scatter([anomaly.data_x[0][int(idx)] for idx in anomaly.anomaly_indices],
                        [anomaly.data_y_orig[0][int(idx)] for idx in anomaly.anomaly_indices],
                        marker='+', s=50, c="red", label='anomaly')

    plt.title('Anomaly Detection in Raman Spectroscopy Data')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
