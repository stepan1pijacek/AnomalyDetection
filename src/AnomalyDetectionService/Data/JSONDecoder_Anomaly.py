import json
import numpy as np

from src.Commons.Models.SpectralAnomalyModel import SpectralAnomalyModel


class JSONDecoder(json.JSONDecoder):
    def __init__(self):
        json.JSONDecoder.__init__(self, object_hook=JSONDecoder.default)

    @staticmethod
    def default(d):
        SpectralAnomalyModel()

        SpectralAnomalyModel.ID = d['ID']
        SpectralAnomalyModel.agent = d['agent']
        SpectralAnomalyModel.iteration = d['iteration']
        SpectralAnomalyModel.concentration = d['concentration']
        SpectralAnomalyModel.data_x = np.asarray(d['data_x'])
        SpectralAnomalyModel.data_y_orig = np.asarray(d['data_y'])
        SpectralAnomalyModel.anomaly_indices = np.asarray([float(i) for i in d['anomaly_indices']])

        return SpectralAnomalyModel
