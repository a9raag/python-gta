import gc
import logging
import sys
import time

import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchmetrics import R2Score, MeanSquaredError

from gta_dataset import GTADataset
from models.model import ResNet, ResidualBlock
from models.test_model import TestModel

logger = logging.getLogger(__name__)


def init_logger():
    logger.setLevel(logging.DEBUG)
    logFormatter = logging.Formatter("%(asctime)s %(filename)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s")
    consoleHandler = logging.StreamHandler(stream=sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    fileHandler = logging.FileHandler(filename="training.log")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)


init_logger()

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using {device} device")


def get_next_layer_dim(I, K, P, S):
    return ((I - K + 2 * P) / S) + 1


import torchvision.transforms as T


def get_train_val_data(batch_size):
    transform = torch.nn.Sequential(T.Resize((224, 224)),
                                    # T.Grayscale(),
                                    T.RandomAdjustSharpness(1.5, p=0.5),
                                    T.RandomErasing(0.3, scale=(0.02, 0.2)),
                                    T.RandomAutocontrast(0.5),
                                    T.RandomPerspective(distortion_scale=0.1, p=0.5)
                                    )
    train_dataset = GTADataset("E:/data/training_data/annotations/v2_balanced_data_train_98k.csv",
                               transform=transform,
                               device=device)
    transform = torch.nn.Sequential(T.Resize((224, 224)),
                                    # T.Grayscale(),
                                    )
    val_dataset = GTADataset("E:/data/training_data/annotations/v2_balanced_data_val_29k.csv",
                             transform=transform,
                             device=device)
    # # generate indices: instead of the actual data we pass in integers instead
    # train_indices, test_indices = train_test_split(
    #     range(len(dataset)),
    #     test_size=test_size,
    #     random_state=seed
    # )

    # generate subset based on indices

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader


def get_model(checkpoint_path=None):
    # model = TestModel()
    # model = ResNet(ResidualBlock, [3, 4, 6, 3])
    model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=1)
    if checkpoint_path is not None:
        logger.info(f"Loading from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)

    model = model.to(device)
    return model


def train():
    logger.info("Model Training Started")
    batch_size = 50

    train_dataloader, val_dataloader = get_train_val_data(batch_size=batch_size)

    num_epochs = 100
    learning_rate = 0.001
    num_classes = 2
    # model = get_model("E:/data/models/res-balanced-v2-22-mse-adam-minmax.pth")
    model = get_model()
    # Loss and optimizer
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    weight_mse = 0.7
    weight_mae = 0.3

    def combined_loss(y_pred, y_true, weight_mse, weight_mae):
        mse = mse_loss(y_pred, y_true)
        mae = mae_loss(y_pred, y_true)
        combined = weight_mse * mse + weight_mae * mae
        return combined

    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=0.00243, momentum=0.5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.1)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    # Model Scoring
    r2score = R2Score(num_outputs=num_classes)
    r2score.to(device)
    mse = MeanSquaredError()
    mse.to(device)

    logger.info(f"Starting model training for {num_epochs} epochs")
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001,
                                            max_lr=0.009,
                                            step_size_up=5,
                                            step_size_down=5,
                                            gamma=0.1,
                                            verbose=True)
    loss = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        train_mse_list = list()
        train_r2_list = list()
        for i, (images, target) in enumerate(train_dataloader):
            # Move tensors to the configured device
            images = images.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            # Forward pass
            preds = model(images)
            loss = combined_loss(preds, target, weight_mse, weight_mae)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), 0.05)
            # scaler.step(optimizer)
            #
            ## Updates the scale for next iteration.
            # scaler.update()

            train_r2_list.append(r2score(preds, target))
            train_mse_list.append(mse(preds, target))
            if i % 1000 == 0:
                logger.info(f"{i}. Training R2={torch.mean(torch.stack(train_r2_list))}")
                logger.info(f"{i}. Training MSE={torch.mean(torch.stack(train_mse_list))}")
                # logger.debug(f"Grads: {[(i, torch.mean(j.grad).tolist()) for i, j in enumerate(model.parameters())]}")
            del images, target, preds
            torch.cuda.empty_cache()
            gc.collect()
        logger.info([(i, torch.mean(j.grad).tolist()) for i, j in enumerate(model.parameters())])
        logger.info(f"Training R2={torch.mean(torch.stack(train_r2_list))}")
        logger.info(f"Training MSE={torch.mean(torch.stack(train_mse_list))}")
        logger.info('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        del train_r2_list, train_mse_list
        torch.cuda.empty_cache()
        gc.collect()
        torch.save(model.state_dict(), f"E:/data/models/resnet18-balanced-v2-{epoch + 1}-l1-CLR-98k-sgd-minmax.pth")

        logger.info(f"Time taken for epoch {time.time() - start_time}")
        scheduler.step()

        # Start Validation
        val_start_time = time.time()
        with torch.no_grad():

            scores = list()
            mse_scores = list()
            validation_loss = list()
            for images, target in val_dataloader:
                images = images.to(device)
                target = target.to(device)
                preds = model(images)
                scores.append(r2score(preds, target))
                mse_scores.append(mse(preds, target))
                validation_loss.append(combined_loss(preds, target, weight_mse, weight_mae))
                # print(f"\tTest R2={scores[-1]},  MSE={mse_scores[-1]}")
                del images, target, preds
            logger.info(f"Validation R2 Score {torch.mean(torch.stack(scores))}")
            logger.info(f"Validation MSE {torch.mean(torch.stack(mse_scores))}")
            logger.info(f"Validation Loss {torch.mean(torch.stack(validation_loss))}")
            del scores, mse_scores, validation_loss
            torch.cuda.empty_cache()
            gc.collect()
        logger.info(f"Time taken for validation {time.time() - val_start_time}")


train()
