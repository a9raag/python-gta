import time

import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
import gc

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# logger.info(f"Using {device} device")


class GTADataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        # scalar = MinMaxScaler(feature_range=(0, 1))
        self.transform = transform
        self.target_transform = target_transform
        if transform is None:
            self.transform = torch.nn.Sequential(T.Resize((224, 224)),
                                                 T.Grayscale(),
                                                 T.ConvertImageDtype(torch.float32))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 5]
        image = read_image(img_path, ImageReadMode.RGB).to(device)
        # image = image / 255.0
        # labels = self.img_labels.iloc[idx].tolist()
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     labels = self.target_transform(labels)
        return image.flatten()


def tomekslinks():
    for i, features1 in enumerate(dataloader):
        features1 = features1.to(device)
        start_time = time.time()
        print(f"Executing batch {i + 1}/{total_batches}")
        start1 = i * batch_size
        end1 = batch_size * (i + 1)
        for j, features2 in enumerate(dataloader):
            start2, end2 = j * batch_size, (j + 1) * batch_size
            features2 = features2.to(device)
            cdist = torch.cdist(features1, features2, p=1)
            for ii in range(cdist.shape[0]):
                for jj in range(cdist.shape[0]):
                    image_pairs.append(start1 + ii, start2 + jj, cdist[ii][[jj]])
            # torch.save(cdist, f"E:/data/training_data/tomeklink/{start1}_{end1}_{start2}_{end2}_cdist.pth")
            del features2, cdist
            torch.cuda.empty_cache()
            gc.collect()
            if j == 1:
                break
        del features1
        torch.cuda.empty_cache()
        gc.collect()

        if i == 1:
            break
        print(f"Time taken to execute batch: {time.time() - start_time}")


dataset = GTADataset("E:/data/training_data/annotations_200k_v2.csv")
batch_size = 5000
dataloader = DataLoader(dataset, batch_size=batch_size)
total_batches = len(dataset) // batch_size
image_pairs = list()
