import numpy as np
import pandas as pd

def tanh(Z):
    # Hyperbolic tangent (tanh) activation function
    return np.tanh(Z)

def gradient_tanh(Z):
    # Gradient of the hyperbolic tangent (tanh) activation function
    return 1 - np.tanh(Z)**2


class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes) + 1

        self._initialise_weights_and_biases(input_size, hidden_sizes, output_size)
    
    def _initialise_weights_and_biases(self, input_size, hidden_sizes, output_size):
        self.weights = []
        self.biases = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, self.num_layers + 1):
            self.weights.append(np.random.randn(sizes[i], sizes[i-1]))
            self.biases.append(np.random.randn(sizes[i], 1))

    def _forward(self, X):
        # Forward pass through the network
        self.activations = [X]
        self.z = []
        for i in range(self.num_layers):
            z = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            self.z.append(z)
            if i < self.num_layers - 1:
                a = tanh(z)  # Tanh activation for hidden layers
            else:
                a = z  # Linear activation for output layer
            self.activations.append(a)
        return self.activations[-1]  # shape: (output_size, m)

    def _backward(self, X, y):
        m = X.shape[1]  # Number of training examples

        # Compute gradients
        gradients = []
        dZ = self.activations[-1] - y  # shape: (output_size, m)
        for i in range(self.num_layers - 1, -1, -1):
            dW = (1 / m) * np.dot(dZ, self.activations[i].T)  # shape: (sizes[i], sizes[i-1])
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)  # shape: (sizes[i], 1)
            gradients.append((dW, db))
            
            if i > 0:
                dA = np.dot(self.weights[i].T, dZ)  # shape: (sizes[i-1], m)
                dZ = dA * gradient_tanh(self.z[i-1])  # shape: (sizes[i-1], m)

        return gradients[::-1]  # Reverse the gradients

    def _update_parameters(self, gradients, learning_rate):
        # Update parameters using gradients and learning rate
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][1]

    def train(self, X_train, y_train, num_epochs: int = 1000, learning_rate: float = 0.01):
        for epoch in range(num_epochs):
            # Forward pass
            outputs = self._forward(X_train.T)

            # Backward pass and parameter update
            gradients = self._backward(X_train.T, y_train.T)
            self._update_parameters(gradients, learning_rate)

            # Compute and print loss
            loss = np.mean((outputs - y_train.T) ** 2)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1} - Loss: {loss}")
    
    def predict(self, X_predict):
        return self._forward(X_predict.T)
