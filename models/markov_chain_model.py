from functools import reduce
from operator import add
from typing import List

import pandas as pd

from models.model import Model


class MarkovChainModel(Model):
    def __init__(self, chain_length: int = 3):
        self.chain_length = chain_length
        self.patterns = {}
        self.pixel_values = {}
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

                chain = self._get_chain_values_behind_point(x, y, self.image_size**0.5, image, self.chain_length)
                chain.insert(0, self.image_size**0.5)
                chain.insert(0, y)
                chain.insert(0, x)

                key = tuple(chain)
                value = image[pix_ix]

                if key in self.patterns: 
                    if value in self.patterns[key]: self.patterns[key][value] += 1
                    else: self.patterns[key][value] = 1
                else: 
                    self.patterns[key] = {}
                    self.patterns[key][value] = 1

    def generate(self, seed: List[int] = None, epochs: int = 3) -> List[int]:
        if seed is None:
            seed = [255]*self.image_size

        if len(seed) != self.image_size:
            raise Exception(f"Seed length does not match expected size: recieved {len(seed)} expected {self.image_size}")

        generated_image = list(seed)
        for _ in range(epochs):
            for pix_ix in range(self.chain_length, self.image_size):
                x = pix_ix%56
                y = pix_ix//56

                chain = self._get_chain_values_behind_point(x, y, self.image_size**0.5, generated_image, self.chain_length)
                chain.insert(0, self.image_size**0.5)
                chain.insert(0, y)
                chain.insert(0, x)
                key = tuple(chain)

                if key in self.patterns:
                    next_val_counts = self.patterns[key]
                    pixel = self._get_pixel_from_likeliness(next_val_counts)

                    generated_image[pix_ix] = pixel
        
        return generated_image