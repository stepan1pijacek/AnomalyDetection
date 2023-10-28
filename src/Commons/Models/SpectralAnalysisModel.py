import numpy as np


class SpectralAnalysisModel:
    def __init__(self):
        self.file_name = str()
        self.ID = str()
        self.agent = str()
        self.concentration = float()
        self.iteration = str()
        self.data_x = np.array([])
        self.data_y = np.array([])