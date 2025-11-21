from functools import reduce
from operator import add
from random import choice
from typing import List

import numpy as np
import pandas as pd
from models.model import Model

from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm

class LongShortTermMemoryModel(Model):
    def __init__(self, window_size: int = 3, is_window_centred: bool = False, hidden_layer_nodes: int = 16):       
        self.window_size = window_size
        self.is_window_centred = is_window_centred
        self.image_size = 0
        self.unique_values = []

        self.rows = []
        self.classes = []

        self.classifier = None
        self.hidden_layer_nodes = hidden_layer_nodes
        self.classifier_metrics = None

    def train(self, df_train: pd.DataFrame, epochs: int = 100, load_file: str = None) -> None:
        df_train = df_train.drop("filename", axis=1)
        image_data = df_train.values.tolist()
        self.image_size = len(image_data[0])

        flattened_image_data = reduce(add, image_data)
        self.unique_values = list(set(flattened_image_data))

        self.rows = []
        self.classes = []

        for image in image_data:
            for pix_ix in range(len(image)):
                x = pix_ix%56
                y = pix_ix//56

                kernel = []
                if self.is_window_centred:
                    kernel = self._get_kernel_values_around_point(x, y, self.image_size**0.5, image, kernel_size=self.window_size, nan_values=-1)
                else:
                    kernel = self._get_kernel_values_behind_point(x, y, self.image_size**0.5, image, kernel_size=self.window_size, nan_values=-1)
                
                key = tuple(kernel)
                value = image[pix_ix]

                self.rows.append(key)
                self.classes.append(value)
        
        columns = ["pix-"+str(i) for i in range(self.window_size**2-1)]
        network_input = pd.DataFrame(self.rows, columns=columns)
        network_input['class'] = self.classes

        X = network_input[columns]
        Y = network_input[['class']]

        X = np.reshape(X, (len(X), self.window_size**2-1, 1))

        X = X / float(4)
        Y = to_categorical(Y)

        model = Sequential()
        model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(Y.shape[1], activation="softmax"))
        if load_file is None:
            model.compile(loss="categorical_crossentropy", optimizer="adam")

            filepath="lstm_model/weights/weights-improvement-{epoch:02d}-{loss:.4f}.keras"
            checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]

            model.fit(X, Y, epochs=epochs, batch_size=128, callbacks=callbacks_list, verbose=1)

        else:
            model.load_weights(load_file)
            model.compile(loss='categorical_crossentropy', optimizer='adam')

        self.classifier = model

    def generate(self, seed: List[int] = None, epochs: int = 1) -> List[int]:
        if seed is None:
            seed = [choice(self.unique_values) for _ in range(self.image_size)]

        if len(seed) != self.image_size:
            raise Exception(f"Seed length does not match expected input size: recieved {len(seed)} expected {self.image_size}")

        generated_image = list(seed)
        for _ in range(epochs):
            for pix_ix in tqdm(range(self.image_size)):
                x = pix_ix%56
                y = pix_ix//56

                kernel_values = []
                if self.is_window_centred:
                    kernel_values = self._get_kernel_values_around_point(x, y, self.image_size**0.5, generated_image, nan_values=-1, kernel_size=self.window_size)
                else:
                    kernel_values = self._get_kernel_values_behind_point(x, y, self.image_size**0.5, generated_image, nan_values=-1, kernel_size=self.window_size)

                key = tuple(kernel_values)
                
                columns = ["pix-"+str(i) for i in range(self.window_size**2-1)]
                df_predict = pd.DataFrame([key], columns=columns)

                prediction = self.classifier.predict(df_predict[columns], verbose=0)
                value = np.argmax(prediction)
                generated_image[pix_ix] = value
            
        return generated_image