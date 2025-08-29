import monai.transforms as mt
import matplotlib.pyplot as plt
import ignite
import numpy as np
import torch
import monai
import warnings

warnings.filterwarnings("ignore")  # remove some scikit-image warnings

monai.config.print_config()

dataset = monai.apps.TransformDataset(
    root_dir="./", task="Task05_Prostate", section="training", transform=None, download=True
)
print(f"\nnumber of subjects: {len(dataset)}")
print(f"The first element in the dataset is\n{dataset[0]}.")
