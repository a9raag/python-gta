import vgamepad as vg

gamepad = vg.VX360Gamepad()

value = int(input())
while True:
    gamepad.left_trigger(value)
    gamepad.right_trigger(value + 10)
    gamepad.update()
