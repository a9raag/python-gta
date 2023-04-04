import random
import torch

from object_detection import plot_one_box
from util.capture_screen import grab_screen
from util.controller import XInputReader
import cv2
import time


def wait_for_q():
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        raise Exception("Quit windows called")


def data_gen():
    xbox = XInputReader()
    while True:
        xbox_input = xbox.read()
        # last_time = time.time()
        screen = grab_screen(region=(0, 40, 800, 500))
        # time_taken = time.time() - last_time

        # print(f'\rFrame took {time_taken} seconds, FPS {1 / time_taken}', end="")
        yield xbox_input, screen
        wait_for_q()


# def get_model():
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#     names = model.names
#     colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
#     return model, names, colors
#
#
# def detect(img):
#     img = cv2.resize(img[:, :, :3], (512, 512))
#     results = model(img)
#     for *xyxy, conf, cls in results.xyxy[0]:
#         label = f'{names[int(cls)]} {conf:.2f}'
#         plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=1)


if __name__ == '__main__':
    try:
        x1, x2, y1, y2 = 100, 350, 150, -150
        for gamepad, img in data_gen():
            text = f"{gamepad.x}, {gamepad.y}, {gamepad.rt}, {gamepad.lt}"
            # cv2.imwrite("../capture.jpg", img)
            coordinates = (10, 100)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 255)
            thickness = 1
            # img = cv2.putText(img, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
            img = img[x1:x2, y1:y2, :3]
            img = cv2.resize(img, (250, 250))
            cv2.imshow("window1", img)
            wait_for_q()
            if cv2.waitKey(1) & 0xFF == ord('x'):
                x1, x2, y1, y2 = list(map(int, input("x1, x2, y1, y2: ").split(",")))
    except Exception as e:
        print(e)
        cv2.destroyAllWindows()
        raise e
