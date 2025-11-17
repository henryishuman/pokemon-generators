from typing import Dict
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def initialise_weights(x, y):
    weights = []
    for _ in range(x * y):
        weights.append(np.random.randn())
    return np.array(weights).reshape(x, y)

class NeuralNetwork:
    def __init__(self, input_count, hidden_layer_count, output_count):
        self.input_weights = initialise_weights(input_count, hidden_layer_count)
        self.hidden_layer_weights = initialise_weights(hidden_layer_count, output_count)

    def train(self, df_train_x: pd.DataFrame, df_train_y: pd.DataFrame, alpha=0.01, epochs=10) -> Dict[str, float]:
        train_x = [df_train_x.to_numpy()]
        train_y = [df_train_y.to_numpy()]

        errors = []
        for _ in range(epochs):
            for i in range(len(train_x)):
                l1 = sigmoid(np.dot(train_x[i], self.input_weights))
                l2 = sigmoid(np.dot(l1, self.hidden_layer_weights))

                error = abs(train_y[i] - l2).mean()
                errors.append(error)

                l2_delta = (train_y[i] - l2)*(l2 *(1-l2))
                self.hidden_layer_weights += l1.T.dot(l2_delta) * alpha

                l1_delta = l2_delta.dot(self.hidden_layer_weights.T) * (l1 * (1-l1))
                self.input_weights += train_x[i].T.dot(l1_delta) * alpha
        
        return {
            "errors": errors,
            "input_weights": self.input_weights,
            "hidden_layer_weights": self.hidden_layer_weights
        }

    def predict(self, df_predict_x: pd.DataFrame) -> float:
        predict_x = np.array(df_predict_x, dtype=float)

        l1 = 1/(1 + np.exp((np.dot(predict_x, self.input_weights))))
        l2 = 1/(1 + np.exp((np.dot(l1, self.hidden_layer_weights))))

        out = np.round(l2,3)
        maxm = 0
        k = 0

        print(out)
        for i in range(len(out[0])):
            if(maxm<out[0][i]):
                maxm = out[0][i]
                k = i
        return k
