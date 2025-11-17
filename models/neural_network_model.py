from functools import reduce
from operator import add
from random import choice
from typing import Any, List

import numpy as np
import pandas as pd
from models.model import Model
from models.model_utils.single_layered_neural_network import SingleLayeredNeuralNetwork


class NeuralNetworkModel(Model):
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

    def train(self, df_train: pd.DataFrame) -> None:
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

        self.classifier = SingleLayeredNeuralNetwork(
            input_count=self.window_size**2-1, 
            hidden_layer_nodes=self.hidden_layer_nodes, 
            output_count=len(self.unique_values)
        )
        self.classifier_metrics = self.classifier.train(network_input[columns], network_input[['class']])

    def generate(self, seed: List[int] = None, epochs: int = 1) -> List[int]:
        if seed is None:
            seed = [choice(self.unique_values) for _ in range(self.image_size)]

        if len(seed) != self.image_size:
            raise Exception(f"Seed length does not match expected input size: recieved {len(seed)} expected {self.image_size}")

        generated_image = list(seed)
        for _ in range(epochs):
            for pix_ix in range(self.image_size):
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

                activated_output_node = self.classifier.predict(df_predict[columns])
                generated_image[pix_ix] = int(self.unique_values[activated_output_node])
            
        return generated_image