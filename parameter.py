# -*- coding: utf-8 -*-
# @Time    : 11/29/21 11:15 AM
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : parameter.py
# @Software: PyCharm

import argparse

parser = argparse.ArgumentParser(description="The parameter of all Algorithms")
parser.add_argument("--height", type=int, default=1000, help="the highest value of the tracking curve")
parser.add_argument("--curve_type", type=str, default="trapezoidal", help="type of curve to be tracked, choose of ['trapezoidal', 'sine']")

parser.add_argument("--dt", type=float, default=0.0001, help="discrete time interval")
parser.add_argument("--start_time", type=float, default=0.0, help="the start time of simulate")
parser.add_argument("--end_time", type=float, default=0.5, help="the end time of simulate")
parser.add_argument("--seed", type=int, default=0, help="the random seed")

parser.add_argument("--change_times", type=int, default=9, help="the change times of system")
parser.add_argument("--use_jump_system", type=bool, default=False, help="select whether to choose a jump system")
parser.add_argument("--used_system_id", type=int, default=1, help="choose the system id when not use jump system. default in [1, 2, 3]")

parser.add_argument("--kp", type=float, default=0.5, help="the kp parameter of PID")
parser.add_argument("--ki", type=float, default=2.985, help="the kp parameter of PID")
parser.add_argument("--kd", type=float, default=0.0, help="the kp parameter of PID")

parser.add_argument("--kp_rl", type=float, default=0.9318, help="the kp parameter of RL Agent Given")
parser.add_argument("--ki_rl", type=float, default=1.4824, help="the ki parameter of RL Agent Given")
parser.add_argument("--kd_rl", type=float, default=0.0, help="the kd parameter of RL Agent Given")

# parameter of DDPG
parser.add_argument("--max_episodes", type=int, default=20000000, help="")
parser.add_argument("--max_ep_steps", type=int, default=4800, help="")
parser.add_argument("--learning_actor", type=float, default=0.0001, help="the learning rate for actor")
parser.add_argument("--learning_critic", type=float, default=0.0001, help="the learning rate for critic")
parser.add_argument("--gamma", type=float, default=0.95, help="the discount of reward")
parser.add_argument("--replace_iter_a", type=int, default=11000, help="the iter steps of replace the actor target network")
parser.add_argument("--replace_iter_c", type=int, default=10000, help="the iter steps of replace the critic target network")
parser.add_argument("--memory_capacity", type=int, default=50000, help="the capacity of the memory replay buffer")
parser.add_argument("--batch_size", type=int, default=256, help="the batch size for training the network")
parser.add_argument("--var", type=float, default=2.0, help="the control exploration of init")
parser.add_argument("--var_min", type=float, default=0.0, help="the control exploration of end")
parser.add_argument("--reg", type=float, default=2.0, help="the regularization parameters")
parser.add_argument("--pid_parameter_search_action_bound", type=float, default=10.0, help="the action bound of pid search for [-10.0, 10.0]")
parser.add_argument("--electric_compensation_action_bound", type=float, default=2.0, help="the action bound of electric search for [-2.0, 2.0]")
parser.add_argument("--state_dim", type=int, default=6, help="the state dim")
parser.add_argument("--pid_parameter_search_action_dim", type=int, default=2, help="the action dim of PID Parameters Adjustment")
parser.add_argument("--electric_compensation_action_dim", type=int, default=1, help="the action dim of electric compensation.")


parser.add_argument("--ep_reward_max", type=int, default=1000000, help="the ")

parser.add_argument("--run_type", type=str, default="train", help="choose from ['train', 'test']")


parser.add_argument("--choose_model", type=str, default='search_electric', help="one of ['search_pid_parameter', 'search_electric', 'class_pid']")


args = parser.parse_args()