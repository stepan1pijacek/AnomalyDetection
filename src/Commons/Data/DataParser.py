import re
import numpy as np

from src.Commons.Models.SpectralAnalysisModel import SpectralAnalysisModel


class DataParser :
    def __init__(self, files = [], folder_path = '', output_folder = '', read_path = ''):
        self.files = files
        self.folder_path = folder_path
        self.output_folder = output_folder
        self.read_path = read_path

    def to_data_model(self) :
        pattern = re.compile("(?P<ID>^.*?(?=-|_))_(?P<Agent>.*?(?=-| )) (?P<Concentration>.*?(?=-|_))_.*_(?P<Iteration>.*?(?=.txt))")
        data_model_list = [SpectralAnalysisModel()]

        for file in self.files :
            data_model = SpectralAnalysisModel()
            match = pattern.match(file)
            file = str(file)
            file_name = re.sub('.txt', '', file)

            if (match != None) :
                data_model.file_name = file_name
                data_model.ID = match.group("ID")
                data_model.agent = match.group("Agent")
                data_model.concentration = match.group("Concentration")
                data_model.iteration = match.group("Iteration")
            else:
                data_model.file_name = file_name
                data_model.ID = None
                data_model.agent = None
                data_model.concentration = 0
                data_model.iteration = None

            with open(self.folder_path+'\\'+file, 'r') as f:
                lines = f.readlines()

                for line in lines:
                    line_data_x, line_data_y = line.split('\t')

                    data_model.data_x = np.append(data_model.data_x, line_data_x.strip())
                    data_model.data_y = np.append(data_model.data_y, float(line_data_y.strip()))
                
                data_model_list.append(data_model)

        return data_model_list