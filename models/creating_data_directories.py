import os
import numpy as np
import math


def create_data_directory(DEST_ROOT, ROOT_DIR, type='train', size=1):
    """
    This function creates the directory for the given type of data, like train, test and validation.
    :param DEST_ROOT: The destination root directory of the data, usually the 'data' directory.
    :param ROOT_DIR: The root directory of the data.
    :param type: The type of data like train, test and validation.
    :param size: The size of the data in the directory like 0.7, 0.2 and 0.1.
    :return: A directory for the given type of data.
    """
    if not os.path.exists(os.path.join(ROOT_DIR)):
        print(f"The source directory {ROOT_DIR} does not exist.")
        return

    number_of_images = {}

    for i in os.listdir(ROOT_DIR):
        number_of_images[i] = len(os.listdir(os.path.join(ROOT_DIR, i)))

    dest_dir = os.path.join(DEST_ROOT, type)

    if not os.path.exists(dest_dir):
        try:
            os.makedirs(dest_dir)
        except PermissionError:
            print(f"Permission denied: Could not create directory {dest_dir}")
            return

        for dir in os.listdir(ROOT_DIR):
            new_dir = os.path.join(dest_dir, dir)

            try:
                os.makedirs(new_dir)
            except PermissionError:
                print(f"Permission denied: Could not create directory {new_dir}")
                return

            images = os.listdir(os.path.join(ROOT_DIR, dir))
            np.random.shuffle(images)

            # Get the number of images for the given type of data as per the size parameter..
            size_dir = math.floor(number_of_images[dir] * size)

            # Move the images to the given type of data directory.
            for image in images[:size_dir]:
                src = os.path.join(ROOT_DIR, dir, image)
                dest = os.path.join(new_dir, image)

                if os.path.exists(src) and not os.path.exists(dest):
                    try:
                        os.rename(src, dest)
                    except PermissionError:
                        print(f"Permission denied: Could not move file {src} to {dest}")
                        return
                else:
                    print(f"File {src} does not exist or file {dest} already exists.")
    else:
        print(f"The directory {type} already exists in {DEST_ROOT}")


# Create the directories for the given type of data.
create_data_directory('data', "Brain Tumor Data Set/Brain Tumor Data Set", "test", 0.15)
print(f"The test directory has been created with {len(os.listdir('data/test'))}.")
create_data_directory('data', "Brain Tumor Data Set/Brain Tumor Data Set", "val", 0.15)
print(f"The val directory has been created with {len(os.listdir('data/val'))}.")
create_data_directory('data',"Brain Tumor Data Set/Brain Tumor Data Set", "train")
print(f"The train directory has been created with {len(os.listdir('data/train'))}.")