import json
import os
import logging
from src.AnomalyDetectionEngine.Data.JSONDecoder import JSONDecoder
from src.AnomalyDetectionEngine.Data.JSONEncoder import JSONEncoder

from src.Commons.Models.DirectoryStructure import DirectoryStructure
from src.Commons.Models.SpectralAnalysisModel import SpectralAnalysisModel
from src.Commons.Settings.GlobalSettings import GlobalSettings


class FileLoader :
    def validate_folder_path (self, folder_path):
        if (not os.path.isdir(folder_path)):
            logging.error("Folder does not exist")
            return False

        logging.info("Folder exists")
        return True

    #TODO: Debug this thing
    def get_files (self, folder_path):
        list_of_files = []

        if (not self.validate_folder_path(folder_path)) :
            return list_of_files

        folders = os.listdir(folder_path)

        logging.info("Geting text and json files")
        for files in folders:
            if (files.endswith(".txt") or files.endswith(".json")):
                list_of_files.append(files)

        return list_of_files

    def get_folders (self, starting_folder_path):
        list_of_folders_paths = []
        list_of_folders_names = []

        folders = os.listdir(starting_folder_path)

        for folder in folders:
            list_of_folders_paths.append(os.path.join(starting_folder_path, folder))
            list_of_folders_names.append(folder)

        return list_of_folders_paths, list_of_folders_names

    def get_nested_files(self, starting_folder_path) :
        paths, names = self.get_folders(starting_folder_path)
        list_of_files = []

        for path in paths :
            directory_structure = DirectoryStructure()
            if(".DS_Store" in path) :
                logging.info("Skipping system folder")
                continue

            folders = os.listdir(path)

            if ((not folders[0].endswith(".txt")) or (not folders[0].endswith(".json")) and (not "kontrola" in path)) :
                for folder in folders :
                    directory_structure.original_folder_name = folder
                    directory_structure.original_folder_path = os.path.join(path, folder)
                    directory_structure.files = self.get_files(directory_structure.original_folder_path)
                    list_of_files.append(directory_structure)
            else :
                file_name = str(path)
                file_name = file_name.split('/')

                directory_structure.original_folder_name = file_name[-1]
                directory_structure.original_folder_path = os.path.join(path)
                directory_structure.files = self.get_files(directory_structure.original_folder_path)
                list_of_files.append(directory_structure)

        return list_of_files

    def get_JSONs(self) :
        data_model_list = []

        files = self.get_files(self.read_path)
        for file in files:
            f = open(os.path.join(self.read_path, file))
            data_model_list.append(json.load(f, cls=JSONDecoder))

        return data_model_list

    def get_JSON(self, file_path) :
        data_model = SpectralAnalysisModel()

        f = open(file_path)
        data_model = json.load(f, cls=JSONDecoder)

        return data_model

    def to_JSON(self, data_model, output_folder) :
        json_string = json.dumps(data_model, cls=JSONEncoder)
        with open(output_folder + "/" + data_model.file_name + '.json', 'w') as outfile:
            outfile.write(json_string)

    def json_data_loader(self):
        control_data = []
        anomalies_data = []
        axis_x = []

        list_of_files_anomalies = self.get_files(GlobalSettings.ANOMALIES_DATA_FILE_PATH)
        list_of_files_control = self.get_files(GlobalSettings.CONTROL_DATA_FILE_PATH)
            
        for file in list_of_files_control:
            json_data = self.get_JSON(os.path.join(GlobalSettings.CONTROL_DATA_FILE_PATH, file))
            control_data.append(json_data.data_y)

        for file in list_of_files_anomalies:
            json_data = self.get_JSON(os.path.join(GlobalSettings.ANOMALIES_DATA_FILE_PATH, file))
            anomalies_data.append(json_data.data_y)
            axis_x.append(json_data.data_x)

        return control_data, anomalies_data, axis_x

    def get_data_json(self, file_path) :
        
        data_y = self.get_JSON(file_path=file_path).data_y 
        data_x = self.get_JSON(file_path).data_x

        return data_y, data_x
