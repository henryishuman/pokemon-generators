import os
from PIL import Image
from typing import List
import numpy as np
import pandas as pd
import imageio

from tqdm import tqdm

from models.bruteforce_model import BruteforceModel
from models.decision_tree_kernel_model import DecisionTreeKernelModel
from models.kernel_model import KernelModel
from models.markov_chain_model import MarkovChainModel
from models.multilayered_perceptron_model import MultilayeredPerceptronModel
from models.neural_network_model import NeuralNetworkModel

def read_images(path_root: str) -> pd.DataFrame:
    filenames = []
    file_data = []

    for file in os.listdir(path_root):
        filename = os.path.join(path_root, file)
        if '.png' in filename:
            img = imageio.imread(filename, mode='F', pilmode='RGB')
            file_data.append(img.flatten()[::3])
            filenames.append(filename)

    img_pixels = ['pix-'+str(i) for i in range(0, 56**2) ]
    df_data = pd.DataFrame(np.array(file_data), columns=img_pixels)
    df_data['filename'] = filenames

    return df_data

def save_image(filepath: str, image_data: List[int]) -> None:
    img = Image.new('RGB', (56, 56), color = 'white')
    pixel_data = img.load()

    for ix in range(len(image_data)-1):
        pixel_data[ix%56, ix//56] = (image_data[ix], image_data[ix], image_data[ix])
    img.save(filepath)

df_images = read_images("img")
if not os.path.exists("generated"):
    os.mkdir("generated")

# width = 56
# chain_length = width // 4
# markov_chain_model = MarkovChainModel(chain_length=chain_length)
# markov_chain_model.train(df_images)

# kernel_sizes = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
# kernel_models = []
# for kernel_size in tqdm(kernel_sizes):
#     kernel_model = KernelModel(kernel_size=kernel_size, is_kernal_centred=kernel_size%2==1)
#     kernel_model.train(df_images)
#     kernel_models.append(kernel_model)

# epochs = 50
# generated_image = markov_chain_model.generate(epochs = epochs)
# save_image("./generated/test_markov.png", generated_image)

# for kernel_model in tqdm(kernel_models):
#     generated_image = kernel_model.generate(epochs = epochs, seed=generated_image)
#     save_image(f"./generated/test_kernel_{kernel_model.kernel_size}.png", generated_image)

window_size = 3
hidden_layer_nodes = [8, 16, 8]
model = MultilayeredPerceptronModel(hidden_layer_nodes=hidden_layer_nodes, window_size=window_size, is_window_centred=True)
model.train(df_images, epochs = 10000, learning_rate=0.001)
generated_image = model.generate()
save_image("./generated/test_nn.png", generated_image)