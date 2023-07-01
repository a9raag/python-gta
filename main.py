import random
import time

import cv2
import numpy as np
import pandas as pd
import pyvjoy
import torch
import torchvision.transforms as T
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

from gta_dataset import MAX_X_VAL, MIN_X_VAL, MAX_TRIGGER_VAL, MIN_TRIGGER_VAL
from models.model import ResidualBlock, ResNet
from models.custom_model import CustomModel
from util.capture_screen import grab_screen
from util.datagen import wait_for_q

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels[self.img_labels['x'] > 40000]
        # self.scaler = MinMaxScaler(feature_range=(-1, 1))
        # self.scaler = self.scaler.fit(self.img_labels[["x", "lt", "rt"]])
        # self.scaled_df = self.scaler.transform(self.img_labels[["x", "lt", "rt"]])
        self.transform = transform
        if transform is None:
            self.transform = torch.nn.Sequential(T.Resize((224, 224)),
                                                 # T.Grayscale(),
                                                 )
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx].image
        image = read_image(img_path, ImageReadMode.RGB)
        labels = [self.img_labels.iloc[idx].x, self.img_labels.iloc[idx].rt]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        return image, torch.Tensor(labels)


# def load_model():
#     resnet = ResNet(ResidualBlock, [3, 4, 6, 3])
#     resnet.load_state_dict(torch.load("E:/data/models/conv-200k_v2-24-mse-adam-minmax.pth"))
#     return resnet


def single_image(row):
    image, labels = row
    image = image.to(device)

    image = resize(image).reshape(-1, 3, 224, 224)
    return image, labels


resize = T.Resize([224, 224])


# dataset = CustomImageDataset("E:/data/training_data/annotations.csv")


def inverse_transform(pred):
    return pred * np.array([MAX_X_VAL - (MIN_X_VAL), MAX_TRIGGER_VAL]) + np.array([MIN_X_VAL, MIN_TRIGGER_VAL])


class VJoyInput:
    def __init__(self):
        self.device = pyvjoy.VJoyDevice(1)

    def get_z_axis(self, lt, rt):
        if lt == rt == 0:
            return 32767 // 2
        z_axis = (lt - rt) * 128 + 32767
        z_axis = max(0, min(z_axis, 32767 * 2))

        return z_axis // 2

    def to_vjoy_metrics(self, x, lt, rt):
        x = (x // 2)
        z = self.get_z_axis(lt, rt)
        return x, z

    def update_controller(self, x, lt, rt):
        x_axis, z_axis = self.to_vjoy_metrics(x, lt, rt)
        self.device.data.wAxisX = x_axis
        self.device.data.wAxisZ = z_axis

        self.device.data.wAxisXRot = 16383
        self.device.data.wAxisYRot = 16383

        self.device.update()  # Send data to vJoy device


def load_model():
    # resnet = TestModel()
    resnet = ResNet(ResidualBlock, [2, 2, 2, 2])
    resnet.load_state_dict(torch.load("E:/data/models/resnet18-balanced-v2-57-l1-CLR-98k-sgd-minmax.pth"))
    return resnet


def run():
    x1, x2, y1, y2 = 100, 350, 150, -150
    model = load_model()
    model = model.to(device)
    model = model.eval()
    with torch.no_grad():
        vjoy = VJoyInput()
        while True:
            screen_og = grab_screen(region=(0, 40, 800, 500))
            screen = cv2.resize(screen_og[x1:x2, y1:y2, :3], (224, 224))
            image = torch.from_numpy(screen) \
                .reshape(-1, 3, 224, 224) \
                .to(device)
            image = image / 255.0
            predictions = model(image)
            output = inverse_transform(predictions.cpu().numpy()[0])
            print(output)
            x, rt = int(output[0]), int(output[1])
            lt = 0
            x -= 16383
            # rt = min(rt, 80)
            # x2 = int(x2 * .75)
            vjoy.update_controller(x, lt, rt)
            # vjoy.update_controller(int(x * random.randrange(50, 80) / 100), lt,
            # #                        int(rt * random.randrange(50, 80) / 100))
            # time.sleep(0.5)q
            time.sleep(0.01)
            # vjoy.update_controller(32767 // 2, 0, 0)
            cv2.imshow("window", screen)
            wait_for_q()


def single_dataset_test():
    annotations_file = "E:/data/training_data/annotations/v2_balanced_data_94k.csv"
    dataset = CustomImageDataset(annotations_file)
    image, labels = single_image(dataset[132])
    image = image / 255.0
    image = image.to(device)
    model = load_model()
    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(image)
        output = inverse_transform(predictions.cpu().numpy()[0])
        x, rt = inverse_transform(output)
        print(x // 2, rt)
        print(output)
        print(labels.cpu().numpy())


if __name__ == '__main__':
    run()
