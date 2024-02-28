import zipfile
import os
import sys
sys.path.append('C:\\Users\\91897\\OneDrive\\Desktop\\PLANT DL MLOPS')
import json
from src.logger.logger import logging
from src.exception.exception import CustomException



class DataIngestion:
    #Config Initialization
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path

    #Loading Config
    def load_config(self):
        with open(self.config_file_path, 'r') as file:
            config = json.load(file)
        return config

    #Create folder for unziping
    def create_folder(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        
    #Unzipping the data
    def unzip_data(self, zip_file_path, extracted_folder_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder_path)
        
    #Extracting the data
    def extract_data(self):
        try:
            
            logging.info("Initiating Data Ingestion")
            config = self.load_config()

            logging.info("Loading Zip File")
            zip_file_path = config["zip_file_path"]
            extracted_folder_path = config["extracted_folder_path"]
            self.create_folder(extracted_folder_path)

            logging.info("Extracting the data from zipped file")
            self.unzip_data(zip_file_path, extracted_folder_path)

            logging.info(f"Data has been successfully extracted to {extracted_folder_path}")

        except Exception as e:
            raise CustomException(e, sys)

    