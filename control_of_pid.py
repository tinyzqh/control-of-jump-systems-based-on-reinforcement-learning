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


def trapezoidal_function(time, args):
    """
    Get the value of the trace curve at the time moment.
    Total simulation time is 0.5s.
    :param time:
    :return:
    """
    height = args.height
    if time <= 0.075:
        return (height / 0.075) * time
    elif 0.075 < time and time <= 0.375:
        return height
    elif 0.375 < time and time <= 0.45:
        return (-height / 0.075) * time + 6 * height
    elif 0.45 < time and time <= 0.5:
        return 0

class PID(object):
    def __init__(self, kp, ki, kd, args):
        super(PID).__init__()
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = args.dt

        self.CumulativeError = 0.0
        self.LastError = None

    def update(self, error):
        p = self.kp * error
        i = self.ki * self.CumulativeError * self.dt
        if self.LastError is None:
            d = 0.0
        else:
            d = self.kd * (error - self.LastError) / self.dt

        self.CumulativeError += error
        self.LastError = error
        return p + i + d


def main(args, times):

    radValues = []
    iaeValues = []
    ecValues = []

    pid = PID(kp=0.5, ki=2.985, kd=0.0, args=args)  # Instance PID Control System.
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