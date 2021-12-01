# -*- coding: utf-8 -*-
# @Time    : 11/30/21 3:17 PM
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : PID.py
# @Software: PyCharm


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
        """
        compute the out of fixed PID parameter.
        :param error:
        :return:
        """
        p = self.kp * error
        i = self.ki * self.CumulativeError * self.dt
        if self.LastError is None:
            d = 0.0
        else:
            d = self.kd * (error - self.LastError) / self.dt

        self.CumulativeError += error
        self.LastError = error
        return p + i + d

    def update_with_parameter(self, error, kp, ki, kd):
        """
        compute the out of the learned PID parameter.
        :param error:
        :param kp:
        :param ki:
        :param kd:
        :return:
        """
        p = kp * error
        i = ki * self.CumulativeError * self.dt
        if self.LastError is None:
            d = 0.0
        else:
            d = kd * (error - self.LastError) / self.dt

        self.CumulativeError += error
        self.LastError = error
        return p + i + d

if __name__ == "__main__":
    pass