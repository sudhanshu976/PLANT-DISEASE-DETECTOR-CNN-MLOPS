
import sys
import os
import tensorflow as tf
sys.path.append('C:\\Users\\91897\\OneDrive\\Desktop\\PLANT DL MLOPS')
from src.logger.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_prepartion import DataPreparation
from src.components.model_trainer import ModelTrainer
from src.components.model_eval import ModelEvaluator


if __name__ == "__main__":

    config_file_path = 'C:\\Users\\91897\\OneDrive\\Desktop\\PLANT DL MLOPS\\src\\config.json'
    ## 1. Data Ingestion

    # Create an instance of DataUnzipper with the config file path
    data_unzipper = DataIngestion(config_file_path)
    # Call the extract_data method to perform the data extraction
    data_unzipper.extract_data()


    ## 2. Data Preparation

    data_preparation = DataPreparation(image_size=256, batch_size=32)
    train_ds, val_ds, test_ds = data_preparation.prepare_datasets("artifacts/unzipped_data/PlantVillage")

    ## 3. Model Trainer
    logging.info("Initiating Model Training")
    input_shape = (32, 256, 256, 3)  # Update with your actual input shape
    n_classes = 3  # Update with your actual number of classes
    model_trainer = ModelTrainer(input_shape=input_shape, n_classes=n_classes)
    model = model_trainer.build_model()
    # Log the model summary
    print(model.summary())
    # Train the model
    epochs = 1
    batch_size = 32
    model_trainer.compile_and_train(model, train_ds, val_ds, epochs, batch_size)
    logging.info("Model Training Completed")

    model_path ="artifacts/saved_models/model.h5"
    model.save(model_path)
    logging.info(f"Model has been saved to {model_path}")

    ## 4. Model Evaluation
    model_evaluator = ModelEvaluator()
    model_evaluator.evaluate_model(model, test_ds)
