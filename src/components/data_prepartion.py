import tensorflow as tf
import sys
import os
sys.path.append('C:\\Users\\91897\\OneDrive\\Desktop\\PLANT DL MLOPS')
from src.logger.logger import logging
from src.exception.exception import CustomException

class DataPreparation:
    #Data initialization
    def __init__(self, image_size=256, batch_size=32, shuffle_size=10000):
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle_size = shuffle_size

    #Load Dataset
    def load_dataset(self, directory):
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            shuffle=True,
            image_size=(self.image_size, self.image_size),
            batch_size=self.batch_size
        )
        self.class_names = dataset.class_names
        return dataset
    
    #Splitting Dataset into Train , Val and Test Sets
    def get_dataset_partition(self, ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True):
        ds_size = len(ds)
        if shuffle:
            ds = ds.shuffle(self.shuffle_size, seed=12)
        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)

        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)

        return train_ds, val_ds, test_ds

    #Prepare datasets for model training
    def prepare_datasets(self, directory):
        try:
            logging.info("Initializing Data Prepartion")
            dataset = self.load_dataset(directory)
            logging.info("Loading Dataset for Data Prepartion")
            train_ds, val_ds, test_ds = self.get_dataset_partition(dataset)
            logging.info("Splitting Data into Train, Validation and Test Sets ")
            
            train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

            logging.info("Cache , Shuffling and Prefetching of Data is Done")
            logging.info("Data Prepartion Completed")

            return train_ds, val_ds, test_ds
            
        except Exception as e:
            raise CustomException(e, sys)


