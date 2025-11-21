from random import choice
from typing import List

import pandas as pd
from tqdm import tqdm

from models.model import Model


class KernelModel(Model):
    def __init__(self, kernel_size: int = 3, is_kernal_centred: bool = False):       
        self.kernel_size = kernel_size
        self.patterns = {}
        self.image_size = 0
        self.unique_values = []
        self.is_kernal_centred = is_kernal_centred

    def train(self, df_train: pd.DataFrame) -> None:
        df_train = df_train.drop("filename", axis=1)

        self.image_size = df_train.iloc[0].size
        self.patterns = {}

        for image_index in tqdm(range(len(df_train)), desc=f"Training kernal model with size {self.kernel_size}..."):
            image = df_train.iloc[image_index].values.tolist()

            for pix_ix in range(len(image)):
                x = pix_ix%56
                y = pix_ix//56
                value = image[pix_ix]
                kernel = []
                if self.is_kernal_centred:
                    kernel = self._get_kernel_values_around_point(x, y, self.image_size**0.5, image)
                else:
                    kernel = self._get_kernel_values_behind_point(x, y, self.image_size**0.5, image)
                
                kernel.insert(0, pix_ix)
                key = tuple(kernel)

                if not value in self.unique_values:
                    self.unique_values.append(value)

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
        for _ in tqdm(range(epochs), desc=f"Generating image from kernel model with size {self.kernel_size}..."):
            for pix_ix in range(self.image_size):
                x = pix_ix%56
                y = pix_ix//56

                kernel_values = []
                if self.is_kernal_centred:
                    kernel_values = self._get_kernel_values_around_point(x, y, self.image_size**0.5, generated_image)
                else:
                    kernel_values = self._get_kernel_values_behind_point(x, y, self.image_size**0.5, generated_image)

                kernel_values.insert(0, pix_ix)
                key = tuple(kernel_values)

                if key in self.patterns:
                    next_val_counts = self.patterns[key]
                    pattern_pixel = self._get_pixel_from_likeliness(next_val_counts)
                    generated_image[pix_ix] = pattern_pixel
            
        return generated_image