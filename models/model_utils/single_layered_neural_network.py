from typing import Dict
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

def sigmoid(x):
    #warnings.filterwarnings('ignore')
    return 1 / (1 + np.exp(-x))

def feed_forward(x, w1, w2):
    x_w1 = x.dot(w1)  
    input_sig = sigmoid(x_w1)
    h_w2 = input_sig.dot(w2)
    hidden_sig = sigmoid(h_w2)
    return hidden_sig

def initialise_weights(x, y):
    weights = []
    for _ in range(x * y):
        weights.append(np.random.randn())
    return np.array(weights).reshape(x, y)

def calculate_loss(out, y):
    s = np.square(out-y)
    s = np.sum(s)/len(y)
    return s 

def back_propagate(x, y, w1, w2, alpha):
    x_w1 = x.dot(w1)  
    input_sig = sigmoid(x_w1)
    h_w2 = input_sig.dot(w2)
    hidden_sig = sigmoid(h_w2)
	
    output_delta = hidden_sig-y
    hidden_delta = np.multiply(w2.dot(output_delta.transpose()).transpose(), np.multiply(input_sig, 1-input_sig))

    w1_adj = x.transpose().dot(hidden_delta)
    w2_adj = input_sig.transpose().dot(output_delta)
	
    w1 = w1-(alpha*(w1_adj))
    w2 = w2-(alpha*(w2_adj))

    return w1, w2

class SingleLayeredNeuralNetwork:
    def __init__(self, input_count, hidden_layer_nodes, output_count):
        self.input_weights = initialise_weights(input_count, hidden_layer_nodes)
        self.hidden_layer_weights = initialise_weights(hidden_layer_nodes, output_count)

    def train(self, df_train_x: pd.DataFrame, df_train_y: pd.DataFrame, alpha=1e-5, epochs=50) -> Dict[str, float]:
        reduced_x = df_train_x / np.max(df_train_x, axis=0)
        reduced_y = df_train_y / np.max(df_train_y, axis=0)

        train_x = [reduced_x.to_numpy()]
        train_y = [reduced_y.to_numpy()]

        acc = []
        loss = []
        for _ in tqdm(range(epochs), desc="training neural network..."):
            l = []
            for i in range(len(train_x)):
                out = feed_forward(
                    train_x[i], 
                    self.input_weights, 
                    self.hidden_layer_weights
                )
                l.append((calculate_loss(out, train_y[i])))

                self.input_weights, self.hidden_layer_weights = back_propagate(
                    train_x[i], train_y[i], 
                    self.input_weights, 
                    self.hidden_layer_weights,
                    alpha
                )

            acc.append((1-(sum(l)/len(train_x)))*100)
            loss.append(sum(l)/len(train_x))
        
        return {
            "accuracy": acc,
            "loss": loss, 
            "input_weights": self.input_weights,
            "hidden_layer_weights": self.hidden_layer_weights
        }

    def predict(self, df_predict_x: pd.DataFrame) -> float:
        reduced_x = df_predict_x / np.max(df_predict_x, axis=0)
        predict_x = reduced_x.to_numpy(dtype=float)

        return feed_forward(
            predict_x, 
            self.input_weights, 
            self.hidden_layer_weights
        )