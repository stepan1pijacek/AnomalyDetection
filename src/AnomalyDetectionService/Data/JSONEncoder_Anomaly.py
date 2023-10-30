import json
import numpy as np

from src.Commons.Models.SpectralAnomalyModel import SpectralAnomalyModel


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, SpectralAnomalyModel):
            return {
                'ID': obj.ID,
                'agent': obj.agent,
                'iteration': obj.iteration,
                'concentration': obj.concentration,
                'data_x': obj.data_x.tolist(),
                'data_y': obj.data_y_orig,
                'anomaly_indices': obj.anomaly_indices,
                'reconstructed_data': obj.reconstructed_data
            }
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)
