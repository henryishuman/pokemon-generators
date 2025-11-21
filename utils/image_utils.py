import os
from PIL import Image, ImageFile, ImageChops, ImageOps
from typing import List

import imageio
import numpy as np
import pandas as pd

from tqdm import tqdm

from random import randrange

def read_images_as_dataframe(path_root: str) -> pd.DataFrame:
    filenames = []
    file_data = []

    for file in tqdm(os.listdir(path_root), desc=f"Loading training data..."):
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

def delete_previous_duplications(path_root: str) -> None:
    duplicate_keys = ["-offset", "-mirrored", "-rotated"]
    for file in os.listdir(path_root):
        filename = os.path.join(path_root, file)
        for key in duplicate_keys:
            if key in filename:
                os.remove(filename)
                break

def create_duplicates_of_images(path_root: str) -> None:
    for file in tqdm(os.listdir(path_root), desc="Duplicating and mutating training data..."): 
        filename = os.path.join(path_root, file)
        if '.png' in filename:
            image = Image.open(filename)
            create_image_duplicates(image, filename.split(".png")[0])

def create_image_duplicates(image: ImageFile, filename_prefix: str) -> None:
    create_offset_images(image, filename_prefix, 20)
    create_mirrored_image(image, filename_prefix)

def create_offset_images(image: ImageFile, filename_prefix: str, count: int, vertical_range: List[int] = [-10, 10], horizontal_range: List[int] = [-10, 10]) -> None:
    for _ in range(count):
        vertical_offset = randrange(vertical_range[0], vertical_range[1])
        horizontal_offset = randrange(horizontal_range[0], horizontal_range[1])

        image_copy = ImageChops.offset(image.copy(), horizontal_offset, vertical_offset)
        image_copy.save(f"{filename_prefix}-offset-({vertical_offset},{horizontal_offset}).png")

def create_mirrored_image(image: ImageFile, filename_prefix: str) -> None:
    mirrored_image = ImageOps.mirror(image.copy())
    mirrored_image.save(f"{filename_prefix}-mirrored.png")
