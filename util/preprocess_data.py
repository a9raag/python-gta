import glob
import numpy as np
import cv2
import pandas as pd

start = 0
data = list()
for file in glob.glob("E:/data/training_data/numpy/*"):
    for x, y, lt, rt, rb, screen in np.load(file, allow_pickle=True):
        image_path = f"E:/data/training_data/images/{start}.jpg"
        cv2.imshow("window", cv2.resize(screen, (224, 224)))
        cv2.waitKey(0)
        # cv2.imwrite(image_path, screen)
        data.append([x, y, lt, rt, rb, image_path])
        break
        start += 1
pd.DataFrame(data).to_csv("E:/data/training_data/images.csv", index=False)
