
from functools import reduce
from operator import add
from random import choice
from typing import List

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from models.model import Model


class DecisionTreeKernelModel(Model):
    def __init__(self, kernel_size: int = 3):       
        self.kernel_size = kernel_size
        self.image_size = 0
        self.unique_values = []
        self.decision_tree_model = None

    def train(self, df_train: pd.DataFrame) -> None:
        df_train = df_train.drop("filename", axis=1)
        image_data = df_train.values.tolist()
        self.image_size = len(image_data[0])

        flattened_image_data = reduce(add, image_data)
        self.unique_values = list(set(flattened_image_data))

        kernels = []
        classes = []
        for image in image_data:
            for pix_ix in range(len(image)):
                x = pix_ix%56
                y = pix_ix//56
                key = self._get_kernel_values_behind_point(x, y, self.image_size**0.5, image)
                value = image[pix_ix]

                kernels.append(key)
                classes.append(value)
                
        columns = ["pix-"+str(i) for i in range(self.kernel_size**2-1)]
        df_train = pd.DataFrame(kernels, columns=columns)
        df_train['class'] = classes
        
        param_grid = {
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        dtree_reg = DecisionTreeRegressor(random_state=42) 
        grid_search = GridSearchCV(estimator=dtree_reg, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0, scoring='neg_mean_squared_error')
        grid_search.fit(df_train[columns], df_train[['class']])
        self.decision_tree_model = grid_search.best_estimator_
        self.decision_tree_model = self.decision_tree_model.fit(df_train[columns], df_train[['class']])
            

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

                key = self._get_kernel_values_behind_point(x, y, self.image_size**0.5, generated_image)
                columns = ["pix-"+str(i) for i in range(self.kernel_size**2-1)]
                df_predict = pd.DataFrame([key], columns=columns)

                pixel = self.decision_tree_model.predict(df_predict)
                generated_image[pix_ix] = int(pixel[0])
            
        return generated_image