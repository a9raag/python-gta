import sys
from ctypes import WinDLL, WinError, Structure, POINTER, byref, c_ubyte
from ctypes.util import find_library
from ctypes.wintypes import DWORD, WORD, SHORT
from collections import namedtuple

# for some reason wintypes.BYTE is defined as signed c_byte and as c_ubyte
BYTE = c_ubyte

# Max number of controllers supported
XUSER_MAX_COUNT = 4


class XINPUT_BUTTONS(Structure):
    """Bit-fields of XINPUT_GAMEPAD wButtons"""

    _fields_ = [
        ("DPAD_UP", WORD, 1),
        ("DPAD_DOWN", WORD, 1),
        ("DPAD_LEFT", WORD, 1),
        ("DPAD_RIGHT", WORD, 1),
        ("START", WORD, 1),
        ("BACK", WORD, 1),
        ("LEFT_THUMB", WORD, 1),
        ("RIGHT_THUMB", WORD, 1),
        ("LEFT_SHOULDER", WORD, 1),
        ("RIGHT_SHOULDER", WORD, 1),
        ("_reserved_1_", WORD, 1),
        ("_reserved_1_", WORD, 1),
        ("A", WORD, 1),
        ("B", WORD, 1),
        ("X", WORD, 1),
        ("Y", WORD, 1)
    ]

    def __repr__(self):
        r = []
        for name, type, size in self._fields_:
            if "reserved" in name:
                continue
            r.append("{}={}".format(name, getattr(self, name)))
        args = ', '.join(r)
        return f"XINPUT_GAMEPAD({args})"


class XINPUT_GAMEPAD(Structure):
    """Describes the current state of the Xbox 360 Controller.

    https://docs.microsoft.com/en-us/windows/win32/api/xinput/ns-xinput-xinput_gamepad

    wButtons is a bitfield describing currently pressed buttons
    """
    _fields_ = [
        ("wButtons", XINPUT_BUTTONS),
        ("bLeftTrigger", BYTE),
        ("bRightTrigger", BYTE),
        ("sThumbLX", SHORT),
        ("sThumbLY", SHORT),
        ("sThumbRX", SHORT),
        ("sThumbRY", SHORT),
    ]

    def __repr__(self):
        r = []
        for name, type in self._fields_:
            r.append("{}={}".format(name, getattr(self, name)))
        args = ', '.join(r)
        return f"XINPUT_GAMEPAD({args})"


class XINPUT_STATE(Structure):
    """Represents the state of a controller.

    https://docs.microsoft.com/en-us/windows/win32/api/xinput/ns-xinput-xinput_state

    dwPacketNumber: State packet number. The packet number indicates whether
        there have been any changes in the state of the controller. If the
        dwPacketNumber member is the same in sequentially returned XINPUT_STATE
        structures, the controller state has not changed.
    """
    _fields_ = [
        ("dwPacketNumber", DWORD),
        ("Gamepad", XINPUT_GAMEPAD)
    ]

    def __repr__(self):
        return f"XINPUT_STATE(dwPacketNumber={self.dwPacketNumber}, Gamepad={self.Gamepad})"


class XInput:
    """Minimal XInput API wrapper"""

    def __init__(self):
        # https://docs.microsoft.com/en-us/windows/win32/xinput/xinput-versions
        # XInput 1.4 is available only on Windows 8+.
        # Older Windows versions are End Of Life anyway.
        lib_name = "XInput1_4.dll"
        lib_path = find_library(lib_name)
        if not lib_path:
            raise Exception(f"Couldn't find {lib_name}")
        self._XInput_ = WinDLL(lib_path)
        self._XInput_.XInputGetState.argtypes = [DWORD, POINTER(XINPUT_STATE)]
        self._XInput_.XInputGetState.restype = DWORD

    def GetState(self, dwUserIndex):
        state = XINPUT_STATE()
        ret = self._XInput_.XInputGetState(dwUserIndex, byref(state))
        if ret:
            raise WinError(ret)
        return state.dwPacketNumber, state.Gamepad


GamePadState = namedtuple("GamepadState", ["x", "y", "lt", "rt", "rb", "start"])


class XInputReader:
    def __init__(self, device=0):
        self.xi = XInput()
        self._detect_controller()
        self.device = device

    def _detect_controller(self):
        for x in range(XUSER_MAX_COUNT):
            try:
                print(f"Reading input from controller {x}")
                print(self.xi.GetState(x))
            except Exception as e:
                print(f"Controller {x} not available: {e}")

    def read(self):
        _i, state = self.xi.GetState(self.device)
        x = state.sThumbLX
        y = state.sThumbLY
        rt = state.bRightTrigger
        lt = state.bLeftTrigger
        rb = state.wButtons.RIGHT_SHOULDER
        start = state.wButtons.START
        return GamePadState(x=x, y=y, lt=lt, rt=rt, rb=rb, start=start)


if __name__ == "__main__":
    xbox = XInputReader(0)
    from time import sleep

    print("Reading all inputs from gamepad 0")
    while True:
        x, y, rt, lt, rb, start = xbox.read()
        print("\r", x, y, lt, rt, rb, start, end="")
        sleep(0.016)
