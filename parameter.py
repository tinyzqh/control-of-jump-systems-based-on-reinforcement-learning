# -*- coding: utf-8 -*-
# @Time    : 11/29/21 11:15 AM
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : parameter.py
# @Software: PyCharm

import argparse

parser = argparse.ArgumentParser(description="The parameter of all Algorithms")
parser.add_argument("--height", type=int, default=2000, help="the highest value of the tracking curve")
parser.add_argument("--dt", type=float, default=0.0001, help="discrete time interval")
parser.add_argument("--start_time", type=float, default=0.0, help="the start time of simulate")
parser.add_argument("--end_time", type=float, default=0.5, help="the end time of simulate")
parser.add_argument("--seed", type=int, default=0, help="the random seed")

parser.add_argument("--change_times", type=int, default=9, help="the change times of system")
parser.add_argument("--use_jump_system", type=bool, default=True, help="select whether to choose a jump system")
parser.add_argument("--used_system_id", type=int, default=1, help="choose the system id when not use jump system. default in [1, 2, 3]")

parser.add_argument("--kp", type=float, default=0.5, help="the kp parameter of PID")
parser.add_argument("--ki", type=float, default=2.985, help="the kp parameter of PID")
parser.add_argument("--kd", type=float, default=0.0, help="the kp parameter of PID")

args = parser.parse_args()