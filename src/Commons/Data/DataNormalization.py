import numpy as np


class DataNormalization:
    def data_normalization(self, data, desired_length):
        # Normalize the test data
        data_norm = [np.pad(data, (0, desired_length - len(data)), mode='constant')]
        data_norm = np.array(data_norm, dtype=np.float32)
        return data_norm
