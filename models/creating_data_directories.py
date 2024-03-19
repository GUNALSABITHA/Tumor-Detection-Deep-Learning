import os
import numpy as np
import math

ROOT_DIR = "Brain Tumor Data Set"

number_of_images = {}

for i in os.listdir(ROOT_DIR):
    number_of_images[i] = len(os.listdir(os.path.join(ROOT_DIR,i)))


if not os.path.exists("data/train"):
    # os.makedirs("data/train")

    for dir in os.listdir(ROOT_DIR):
        # print(dir)
        # os.makedirs(os.path.join(f"data/train/{dir}"))
        #
        print(os.path.join(ROOT_DIR,dir))
        # for img in np.random.choice(
        #     a = os.listdir(os.path.join(ROOT_DIR,dir)),
        #     size = int(math.floor(0.7*number_of_images[dir])),
        #     replace=False
        # ):
        #     OriginalPath = os.path.join(ROOT_DIR,dir,img)
        #     DestinationPath = os.path.join("data/train",dir)
        #     print(f"Copying {OriginalPath} to {DestinationPath}")
        #     os.system(f"cp {OriginalPath} {DestinationPath}")
        #     # os.remove(OriginalPath)