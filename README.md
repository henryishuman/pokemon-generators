# Pokémon Sprite Generators

Various pipelines and models that are used to attempt to generate an image of Pokémon.

## Installation

Before attempting to run anything, make sure that all required python packages have been installed by using the following command:

`pip install -r requirements.txt`

## Structure

This project is organised into `models` and `pipelines`. 

`Models` are trained on a `pandas` `DataFrame` containing a list of images represented by four pixel values, each being the colour of said pixel, i.e. `[0, 85, 170, 255]` encoding the colours black, dark grey, light grey, and white.

`Pipelines` are used to build up a series of reads, data transformations, model training, and generation of sprite images.

## Running

So far, the only pipeline which exists is that of the `kernal_model_pipeline`. I am planning on adding further pipelines in the future. To run this, execute the `generate_pokemon_images.py` script, i.e.:

`python generate_pokemon_images.py`