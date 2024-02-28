import sys
import os
sys.path.append('C:\\Users\\91897\\OneDrive\\Desktop\\PLANT DL MLOPS')
import tensorflow as tf
from src.logger.logger import logging
from src.exception.exception import CustomException

class ModelEvaluator:
    def __init__(self):
        pass

    # Evaluate Model
    def evaluate_model(self,model, test_ds):
        try:
            logging.info("Initiating Model Evaluation on Test Set")

            # Evaluate the model
            scores = model.evaluate(test_ds)
            logging.info(f"Loss: {scores[0]}, Accuracy: {scores[1]}")
            logging.info("Model Evaluation Completed")

        except Exception as e:
            error_message = f"Error during model evaluation: {str(e)}"
            logging.error(error_message)
            raise CustomException(error_message, sys)

