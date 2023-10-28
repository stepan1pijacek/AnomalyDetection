import numpy as np

from src.AnomalyDetectionEngine.Data.Filter.DataPreProcessor import Filter as rc
from src.AnomalyDetectionEngine.Utilities.FileLoader import FileLoader as fl
from src.Commons.Models.SpectralAnalysisModel import SpectralAnalysisModel
from src.Commons.Data.DataParser import DataParser as dp
from src.Commons.Settings.GlobalSettings import GlobalSettings


class DataProcessing:
    def data_raw_to_clean_from_files(self, output_folder, data_file_list):
        file_loader = fl()
        data_preprocesor = rc()

        model_list = [SpectralAnalysisModel()]
        data_model_list = [SpectralAnalysisModel()]

        for files in data_file_list:
            files_to_process = file_loader.get_files(files)
            data = dp(files_to_process, files,
                      output_folder, output_folder)
            data_model_list.append(data.to_data_model())

        data_model_list.pop(0)
        for model in data_model_list:
            for sam in model:
                if (sam.ID == "" and sam.file_name == ""):
                    continue

                filtered_data = data_preprocesor.clean_data(np.array(sam.data_x), np.array(sam.data_y),
                                                            GlobalSettings.BALL_RADIUS,
                                                            GlobalSettings.PASSES, GlobalSettings.PLOTTING,
                                                            GlobalSettings.NORMALIZATION_PEAK,
                                                            GlobalSettings.WINDOW_LENGTH, GlobalSettings.POLYORDER)
                filtered_data = np.array(filtered_data.tolist())
                sam.data_y = filtered_data
                model_list.append(sam)

        model_list.pop(0)

        for model in model_list:
            file_loader.to_JSON(model, output_folder)

    def data_raw_to_clean_from_models(self, model_list):
        data_preprocesor = rc()
        processed_data = [SpectralAnalysisModel()]

        model_list.pop(0)
        for model in model_list:

            if model.ID == "" and model.file_name == "":
                continue

            filtered_data = data_preprocesor.clean_data(np.array(model.data_x), np.array(model.data_y),
                                                            GlobalSettings.BALL_RADIUS,
                                                            GlobalSettings.PASSES, GlobalSettings.PLOTTING,
                                                            GlobalSettings.NORMALIZATION_PEAK,
                                                            GlobalSettings.WINDOW_LENGTH, GlobalSettings.POLYORDER)
            filtered_data = np.array(filtered_data.tolist())
            model.data_y = filtered_data
            processed_data.append(model)

        processed_data.pop(0)
        return processed_data
