import tensorflow as tf
import sys
import os
sys.path.append('C:\\Users\\91897\\OneDrive\\Desktop\\PLANT DL MLOPS')
from tensorflow.keras import layers, models
from src.components.data_prepartion import DataPreparation
from src.components.model_eval import ModelEvaluator
from src.logger.logger import logging

class ModelTrainer:
    def __init__(self, input_shape, n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes

    #Model architecture
    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape[1:])  # Exclude batch size from input shape

        # Preprocessing layers
        x = layers.Resizing(self.input_shape[1], self.input_shape[2])(inputs)
        x = layers.Rescaling(1.0 / 255)(x)
        x = layers.RandomFlip("horizontal_and_vertical")(x)
        x = layers.RandomRotation(0.2)(x)

        # Convolutional layers
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Fully connected layers
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    #Compiling an Training model
    def compile_and_train(self, model, train_ds, val_ds, epochs, batch_size):
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        history = model.fit(
            train_ds,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_data=val_ds
        )
        logging.info(history.history)
        return history
    
    #Saving model
    def save_model(self, given_model, model_path):
        logging.info(f"Saving the model to {model_path}")
        model=given_model
        model.save(model_path)
        

