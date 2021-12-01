# -*- coding: utf-8 -*-
# @Time    : 11/29/21 11:13 AM
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : control_of_pid.py
# @Software: PyCharm

import numpy as np
import seaborn as sns
from parameter import args
from servo_system import ServoSystem
import matplotlib.pyplot as plt
from PID import PID
from utils import trapezoidal_function

def main(args, times):

    radValues = []
    iaeValues = []
    ecValues = []

    pid = PID(kp=args.kp, ki=args.ki, kd=args.kd, args=args)  # Instance PID Control System.
    servo_system = ServoSystem()  # Instance Servo System.

    for sim_time in times:
        error = trapezoidal_function(time=sim_time, args=args) - servo_system.rad_cur

        ec_cur = pid.update(error=error)

        servo_system.step(action=ec_cur)

        # append the value
        radValues.append(servo_system.rad_cur)
        iaeValues.append(abs(error) if len(iaeValues) == 0 else abs(error) + iaeValues[-1])
        ecValues.append(ec_cur)

    return radValues, iaeValues, ecValues


if __name__ == "__main__":
    times = np.arange(args.start_time, args.end_time, args.dt)
    SpeedCommand = [trapezoidal_function(t, args) for t in times]
    radValues, iaeValues, ecValues = main(args=args, times=times)

    sns.set(style="darkgrid", font_scale=1.5, font='serif', rc={'figure.figsize': (10, 8)})
    plt.subplot(2, 2, 1)
    plt.plot(times, SpeedCommand, label="Speed Command")
    plt.plot(times, radValues, label = "Class PID Respond")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed Command (rmp)")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(times, iaeValues, label="Classic PID")
    plt.xlabel("Time (s)")
    plt.ylabel("IAE")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(times, ecValues, label="Classic PID")
    plt.xlabel("Time (s)")
    plt.ylabel("Electric (A)")
    plt.legend()

    plt.show()