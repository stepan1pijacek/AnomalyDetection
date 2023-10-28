import numpy as np


class SpectralAnomalyModel:
    def __int__(self):
        self.ID = str()
        self.agent = str()
        self.iteration = str()
        self.concentration = float()
        self.data_x = np.array([])
        self.data_y_orig = np.array([])
        self.anomaly_indices = np.array([])
