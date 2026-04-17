import pyspacemouse
import time
from piper_sdk import *

if __name__ == "__main__":
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    while( not piper.EnablePiper()):
        time.sleep(0.01)
    piper.GripperCtrl(0, 1000, 0x01, 0)
    
    factor = 1000

    X = 300.614
    Y = -12.185
    Z = 282.341
    RX = -179.351
    RY = 23.933
    RZ = 177.934

    X = round(X*factor)
    Y = round(Y*factor)
    Z = round(Z*factor)
    RX = round(RX*factor)
    RY = round(RY*factor)
    RZ = round(RZ*factor)
    joint_6 = round(0.08*1000*1000)

    piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
    piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)
    piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)

    with pyspacemouse.open() as device:
        while True:
            state = device.read()
            # print(state)
            state_X = round(state.x*factor)
            state_Y = round(state.y*factor)
            state_Z = round(state.z*factor)

            X = round(X + state_X)
            Y = round(Y + state_Y)
            Z = round(Z + state_Z)
            RX = round(RX + state.roll*factor)
            RY = round(RY + state.pitch*factor)    
            RZ = round(RZ + state.yaw*factor)

            piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
            piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)

            if state.buttons[0]: 
                joint_6 = round(0.08*1000*1000)
            elif state.buttons[1]: 
                joint_6 = round(0.00*1000*1000)
            piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)

            time.sleep(0.01)
    