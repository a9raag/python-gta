import os.path
import time

import cv2
import pandas as pd

from util.datagen import data_gen

ANNOTATIONS_DIR = "E:/data/training_data/annotations/v2"
IMAGES_DIR = "E:/data/training_data/images/v2"
CHECKPOINT_DIR = "D:/Dev/self_driving_gta/checkpoint_v2"


class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_file = "LATEST"
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file)
        self.latest_path = self._get_latest()

    def get_latest(self):
        return self.latest_path

    def _get_latest(self):
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, mode="r") as file:
                checkpoint_path = file.read()
                if os.path.exists(checkpoint_path):
                    return checkpoint_path
        return None

    def create(self, filename):
        print("Creating checkpoint for batch ", filename)
        with open(self.checkpoint_path, mode="w") as file:
            file.write(filename)


class DataManager:
    def __init__(self, images_dir, annotations_dir, checkpoint_dir):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        os.makedirs(annotations_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        self.checkpoint = CheckpointManager(checkpoint_dir)

    def _cleanup_images(self, batch):
        print("Start: Image Cleanup")
        for x, y, lt, rt, rb, img in batch:
            if os.path.exists(img):
                print("\tRemoving image ", img, "FOUND")
                os.remove(img)
            else:
                print(img, "NOT FOUND")

    def save_batch(self, batch, filepath=None):
        try:
            if filepath is None:
                filepath = os.path.join(self.annotations_dir, f"{int(time.time())}.csv")
            print(f"saving batch of length {len(batch)} to file: {filepath}")
            pd.DataFrame(batch).to_csv(filepath, index=False, header=["x", "y", "lt", "rt", "rb", "image"])
            self.checkpoint.create(filepath)
        except Exception as e:
            print("Failed to save batch", e)
            self._cleanup_images(batch)
            raise e

    def save_img(self, img, filename=None):
        if filename is None:
            filename = f"{int(time.time() * 1000)}.jpg"
        path = os.path.join(self.images_dir, filename)
        cv2.imwrite(path, img)
        return path

    def recent_batch(self):
        batch = list()
        latest_checkpoint = self.checkpoint.get_latest()
        if latest_checkpoint is not None:
            batch = pd.read_csv(latest_checkpoint).values.tolist()
            print(f"Checkpoint exists reading from checkpoint, with {len(batch)} records")
        return batch, latest_checkpoint


def add_text_to_img(img, gamepad):
    new_img = img.copy()
    text = f"x={gamepad.x}, y={gamepad.y}, lt={gamepad.lt}, rt={gamepad.rt}"
    # cv2.imwrite("../capture.jpg", img)
    coordinates = (300, 20)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255, 0, 255)
    thickness = 1
    return cv2.putText(new_img, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)


def collect():
    batch_size = 1000
    x1, x2, y1, y2 = 100, 350, 150, -150
    dm = DataManager(IMAGES_DIR, ANNOTATIONS_DIR, CHECKPOINT_DIR)
    training_batch, recent_checkpoint = dm.recent_batch()
    time.sleep(10)
    total_processed = 0
    last_paused = time.time()
    paused = False
    try:
        for gamepad, screen in data_gen():
            if gamepad.start == 1 and time.time() - last_paused > 0.6:
                last_paused = time.time()
                paused = not paused
            if paused:
                continue
            if len(training_batch) == batch_size:
                print()
                # update existing batch file if most recent batch was incomplete
                dm.save_batch(training_batch, recent_checkpoint)
                # set `recent_batch` to None to create new batch files
                recent_checkpoint = None
                training_batch = list()
            x, y, lt, rt, rb = gamepad.x, gamepad.y, gamepad.lt, gamepad.rt, gamepad.rb
            screen = screen[x1:x2, y1:y2, :3]
            # cv2.imshow('GTA Capture', add_text_to_img(screen, gamepad))
            screen = cv2.resize(screen, (250, 250))
            cv2.imshow('GTA Capture Resize', add_text_to_img(screen, gamepad))
            image_path = dm.save_img(screen)
            training_batch.append([x, y, lt, rt, rb, image_path])
            total_processed += 1
            print("\r", len(training_batch), total_processed, end="")

    except Exception as e:
        if len(training_batch) > 0:
            print("Saving incomplete batch")
            dm.save_batch(training_batch, recent_checkpoint)
        raise e


if __name__ == '__main__':
    collect()
