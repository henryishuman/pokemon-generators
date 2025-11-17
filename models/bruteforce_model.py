from functools import reduce
from operator import add
from random import choice
from typing import List

import pandas as pd

from models.model import Model


class BruteforceModel(Model):
    def __init__(self):       
        self.patterns = {}
        self.image_size = 0
        self.unique_values = []

    def train(self, df_train: pd.DataFrame) -> None:
        df_train = df_train.drop("filename", axis=1)
        image_data = df_train.values.tolist()
        self.image_size = len(image_data[0])

        flattened_image_data = reduce(add, image_data)
        self.unique_values = list(set(flattened_image_data))

        self.patterns = {}

        for image in image_data:
            for pix_ix in range(len(image)):
                x = pix_ix%56
                y = pix_ix//56
                value = image[pix_ix]
                kernel = self._get_all_values_behind_point(x, y, self.image_size**0.5, image)

                key = tuple(kernel)

                if key in self.patterns:
                    if value in self.patterns[key]: self.patterns[key][value] += 1
                    else: self.patterns[key][value] = 1
                else:
                    self.patterns[key] = {}
                    self.patterns[key][value] = 1            

    def generate(self, seed: List[int] = None, epochs: int = 10) -> List[int]:
        if seed is None:
            seed = [choice(self.unique_values) for _ in range(self.image_size)]

        if len(seed) != self.image_size:
            raise Exception(f"Seed length does not match expected input size: recieved {len(seed)} expected {self.image_size}")

        generated_image = list(seed)
        for _ in range(epochs):
            for pix_ix in range(self.image_size):
                x = pix_ix%56
                y = pix_ix//56

                kernel_values = self._get_all_values_behind_point(x, y, self.image_size**0.5, generated_image)
                key = tuple(kernel_values)

                if key in self.patterns:
                    next_val_counts = self.patterns[key]
                    pattern_pixel = self._get_pixel_from_likeliness(next_val_counts)
                    generated_image[pix_ix] = pattern_pixel
                else:
                    pixel = int(choice(self.unique_values))
                    generated_image[pix_ix] = pixel
            
        return generated_image