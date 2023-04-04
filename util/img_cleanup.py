import os

import pandas as pd
import cv2


def wait_for_q():
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        raise Exception("Quit windows called")


def delete_data(annotation_file):
    df = pd.read_csv(annotation_file)
    for t in df.itertuples():
        if os.path.exists(t.image):
            os.remove(t.image)
    os.remove(annotation_file)


def view_data(annotation_file):
    df = pd.read_csv(annotation_file)
    for t in df.itertuples():
        if os.path.exists(t.image):
            image = cv2.imread(t.image)
            cv2.imshow("cleanup_window", image)
        wait_for_q()


annotation_files = "E:/data/training_data/annotations/1676713234.csv"
view_data(annotation_files)
