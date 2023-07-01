import numpy as np
import pandas as pd
import torch
import cv2

from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

MAX_X_VAL = 65534
MIN_X_VAL = 0
MIN_TRIGGER_VAL = 0
MAX_TRIGGER_VAL = 255


class GTADataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None, device=None):
        self.device = device
        self.df = pd.read_csv(annotations_file)
        print("Dataset size", self.df.shape[0])
        # scalar = MinMaxScaler(feature_range=(0, 1))
        #
        # self.scaled_df = (self.df[["x", "rt"]] - np.array([MIN_X_VAL, MIN_TRIGGER_VAL])) / np.array(
        #     [MAX_X_VAL - MIN_X_VAL, MAX_TRIGGER_VAL - MIN_TRIGGER_VAL])

        self.scaled_df = self.df[["x", "rt"]]
        self.target_transform = target_transform
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def preprocessing(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx].image
        try:
            image = read_image(img_path, ImageReadMode.RGB)
            if self.transform:
                image = self.transform(image)
            labels = self.scaled_df.iloc[idx].values
            if self.target_transform:
                labels = self.target_transform(labels)

            image = image / 255.0
            labels = torch.Tensor(labels)
            if self.device is not None:
                image = image.to(self.device)
                labels = labels.to(self.device)
            return image, labels
        except Exception as e:
            print("Failed to load image", img_path)
            raise e
