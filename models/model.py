from abc import abstractmethod
from math import floor
from typing import Any, Dict, List

import pandas as pd
from numpy import array
from numpy.random import choice


class Model:
    @abstractmethod
    def train(df_train: pd.DataFrame) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def generate(self, seed: List[int] = None, epochs: int = 10) -> List[int]:
        raise NotImplementedError
    
    def _get_pixel_from_likeliness(self, value_counts: Dict[int, int]) -> int:
        values = list(value_counts.values())
        keys = list(value_counts.keys())
        exp_probabilities = (array(values)/sum(values))
        probabilities = exp_probabilities/sum(exp_probabilities)
        val = choice(keys, p=probabilities) 
        return int(val)
    
    def _get_all_values_behind_point(self, x: int, y: int, width: int, full_image: List[int]) -> Dict[int, int]:
        current_pix_ix = int(x + y * width)
        
        values = [
            int(full_image[pix_ix])
            for pix_ix in range(current_pix_ix)
        ]
        return values
    
    def _get_chain_values_behind_point(self, x: int, y: int, width: int, full_image: List[int], chain_size: int = 3, nan_values: Any = None) -> Dict[int, int]:
        pix_ix = int(y * width + x)

        get_x = lambda pix_ix: int(pix_ix % width)

        kernel_values = [
            int(full_image[pix_ix - pix_mod])
            if (pix_ix - pix_mod >= 0) and (get_x(pix_ix - pix_mod) < width)
            else nan_values
            for pix_mod in range(chain_size, 0, -1)
        ]
        return kernel_values
    
    def _get_kernel_values_around_point(self, x: int, y: int, width: int, full_image: List[int], kernel_size: int = 3, nan_values: Any = None) -> Dict[int, int]:
        kernel_min = -floor(kernel_size/2)
        kernel_max = floor(kernel_size/2)+1
        pix_ix = x + y*width

        kernel_values = [
            int(full_image[int((x+dx) + (y+dy)*width)])
            if ((x+dx) >= 0 and (y+dy) >= 0) and 
                ((x+dx) < width and (y+dy) < width)
            else nan_values
            for dy in range(kernel_min, kernel_max) 
            for dx in range(kernel_min, kernel_max)
            if (pix_ix != (x+dx) + (y+dy)*width)
        ]
        return kernel_values

    def _get_kernel_values_behind_point(self, x: int, y: int, width: int, full_image: List[int], kernel_size: int = 3, nan_values: Any = None) -> Dict[int, int]:
        kernel_values = [
            int(full_image[int((x+dx) + (y+dy)*width)])
            if ((x+dx) >= 0 and (y+dy) >= 0) and 
                ((x+dx) < width and (y+dy) < width)
            else nan_values
            for dy in range(1-kernel_size,1)
            for dx in range(1-kernel_size,1) 
            if dx + dy != 0
        ]
        return kernel_values

    def _get_distances_from_edge_of_image(self, x: int, y: int, width: int, height: int) -> List[int]:
        t_distance = y
        l_distance = x
        b_distance = height - y
        r_distance = width - x
        return [t_distance, r_distance, b_distance, l_distance]