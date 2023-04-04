import random
import pyvjoy
import time

from util.controller import XInputReader


class VJoyInput:
    def __init__(self):
        self.device = pyvjoy.VJoyDevice(1)

    def update_controller(self, output):
        self.device.data.wAxisX = output[0]
        self.device.data.wAxisZ = output[1]

        self.device.data.wAxisXRot = 16383
        self.device.data.wAxisYRot = 16383

        self.device.update()  # Send data to vJoy device


def get_z_axis(lt, rt):
    if lt == rt == 0:
        return 32767 // 2
    z_axis = (lt - rt) * 128 + 32767
    z_axis = max(0, min(z_axis, 32767 * 2))

    return z_axis // 2


def to_vjoy_metrics(x, lt, rt):
    x = x + 32767 // 2
    z = get_z_axis(lt, rt)
    return x, z


def compare_with_xbox():
    vjoy = VJoyInput()
    xbox = XInputReader()
    while True:
        time.sleep(0.15)
        gamepad = xbox.read()
        x = gamepad.x + 32767 // 2
        lt = gamepad.lt
        rt = gamepad.rt
        z = get_z_axis(lt, rt)
        print("\r", x, lt, rt, z, end="")
        vjoy.update_controller([x, z])


def random_input():
    vjoy = VJoyInput()
    while True:
        time.sleep(0.15)
        x = random.randrange(0, 32767)
        lt = random.randrange(0, 255)
        rt = random.randrange(0, 255)
        z = get_z_axis(lt, rt)
        print("\r", x, lt, rt, z, end="")
        vjoy.update_controller([x, z])


if __name__ == '__main__':
    random_input()
# input("Continue?")
