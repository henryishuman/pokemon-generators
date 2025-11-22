import os

from models.kernel_model import KernelModel
from utils.image_utils import create_duplicates_of_images, delete_previous_duplications, read_images_as_dataframe, save_image


def run_pipeline(image_root_path: str, output_directory: str = "generated") -> None:
    print("Duplicating images for training...")
    delete_previous_duplications(image_root_path)
    create_duplicates_of_images(image_root_path)

    print("Reading images as integer data...")
    df_images = read_images_as_dataframe(image_root_path)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    print("Training kernel models...")
    kernel_sizes = [1, 2, 3, 4, 5, 7, 11, 13, 11, 7, 5, 4, 3, 2, 1]
    kernel_models = []
    for kernel_size in kernel_sizes:
        kernel_model = KernelModel(kernel_size=kernel_size, is_kernal_centred=kernel_size%2==1)
        kernel_model.train(df_images)
        kernel_models.append(kernel_model)

    print("Generating images from kernel models...")
    generated_image = None
    for index, kernel_model in enumerate(kernel_models):
        generated_image = kernel_model.generate(epochs = 100, seed=generated_image)
        save_image(f"./{output_directory}/test_kernel_{index}_size_{kernel_model.kernel_size}.png", generated_image)

    print("Cleaning up temporary data...")
    delete_previous_duplications(image_root_path)

    print("Complete!")