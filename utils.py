# -*- coding: utf-8 -*-
# @Time    : 11/30/21 3:21 PM
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : utils.py
# @Software: PyCharm

from parameter import args

def trapezoidal_function(time):
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

if __name__ == "__main__":
    pass