from functools import reduce
from operator import add
from typing import List

import pandas as pd

from models.model import Model

from random import choice


class CellPositionModel(Model):
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
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

                window = self._get_pixel_values_after_point(x, y, self.window_size, self.image_size**0.5, image)

                key = (x, y)
                value = tuple(window)

                if key in self.patterns: 
                    if value in self.patterns[key]: self.patterns[key][value] += 1
                    else: self.patterns[key][value] = 1
                else: 
                    self.patterns[key] = {}
                    self.patterns[key][value] = 1

    def generate(self, seed: List[int] = None, epochs: int = 3) -> List[int]:
        if seed is None:
            seed = [choice(self.unique_values) for _ in range(self.image_size)]

        if len(seed) != self.image_size:
            raise Exception(f"Seed length does not match expected size: recieved {len(seed)} expected {self.image_size}")

        generated_image = list(seed)
        for _ in range(epochs):
            for pix_ix in range(self.image_size):
                x = pix_ix%56
                y = pix_ix//56

                key = (x, y)

                if key in self.patterns:
                    next_val_counts = self.patterns[key]
                    pixels = self._get_pixels_from_likeliness(next_val_counts)

                    for i, pix in enumerate(pixels):
                        if pix_ix + i < self.image_size:
                            generated_image[pix_ix + i] = pix
        
        return generated_image