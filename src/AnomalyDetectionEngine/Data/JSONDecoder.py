import json
import numpy as np

from src.Commons.Models.SpectralAnalysisModel import SpectralAnalysisModel


class JSONDecoder(json.JSONDecoder):
    def __init__(self):
        json.JSONDecoder.__init__(self, object_hook=JSONDecoder.default)
    
    @staticmethod
    def default(d):
        SpectralAnalysisModel()

        if d.get('file_name') != None:
            SpectralAnalysisModel.file_name = d['file_name']
            SpectralAnalysisModel.ID = d['ID']
            SpectralAnalysisModel.agent = d['agent']
            SpectralAnalysisModel.iteration = d['iteration']
            SpectralAnalysisModel.concentration = d['concentration']
            # How to convert a list of strings to a numpy array of floats
            # https://stackoverflow.com/questions/4663306/how-to-convert-a-list-of-strings-to-a-numpy-array-of-floats
            SpectralAnalysisModel.data_x = np.asarray([float(i) for i in d['data_x']])
            SpectralAnalysisModel.data_y = np.asarray([float(i) for i in d['data_y']])

            return SpectralAnalysisModel
        return d