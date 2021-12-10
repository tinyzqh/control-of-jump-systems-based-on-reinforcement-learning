# -*- coding: utf-8 -*-
# @Time    : 11/30/21 2:34 PM
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : servo_system_env.py
# @Software: PyCharm


from parameter import args
from utils import trapezoidal_function
from PID import PID
import numpy as np
import gym


class ServoSystemEnv(gym.Env):
	def __init__(self):
		super(ServoSystemEnv, self).__init__()

		self.rad_last = 0.0  # Speed at the previous moment
		self.rad_cur = 0.0  # Speed at the moment
		self.rad_next = 0.0  # Speed at the next moment
		self.rad_next_next = 0.0  # Speed at the next next moment
		self.error_last = 0.0  # The error at the previous moment
		self.error_cur = 0.0  # The error at the moment

		self.ec_last = 0.0  # electric current at the previous moment
		self.cnt = 0  # used to count the step number

		self.rad_threshold = 9000

		self.times = np.arange(args.start_time, args.end_time, args.dt)  # generate the time steps.
		np.random.seed(args.seed)
		self.SystemsList = np.random.randint(0, 3, args.change_times)  # Generate the random systems id.
		self.SystemsLists = [i for i in self.SystemsList for j in range(int(len(self.times)/args.change_times) + 1)] # extended system id.

		self.state = np.array([self.rad_last, self.rad_cur, self.rad_next, self.rad_next_next, self.error_last, self.error_cur])


		if args.choose_model == "search_pid_parameter":
			self.pid = PID(kp=args.kp, ki=args.ki, kd=args.kd, args=args)
		else:
			self.pid = PID(kp=args.kp_rl, ki=args.ki_rl, kd=args.kd_rl, args=args)

	def update_rad(self, electric):
		"""
		Update the rad value Based on different Parameters.
		:param electric:
		:return:
		"""
		if args.use_jump_system:
			if self.SystemsLists[self.cnt] == 0:
				rad = self._system1(ec_cur = electric)
			elif self.SystemsLists[self.cnt] == 1:
				rad = self._system2(ec_cur = electric)
			else:
				assert self.SystemsLists[self.cnt] == 2, print("the system id of SystemsLists is out of range.")
				rad = self._system3(ec_cur = electric)
		else:
			if args.used_system_id == 1:
				rad = self._system1(ec_cur = electric)
			elif args.used_system_id == 2:
				rad = self._system2(ec_cur = electric)
			else:
				assert args.used_system_id == 3, print("the selected system id is illegal")
				rad = self._system2(ec_cur = electric)

		return rad

	def step(self, action):
		"""
		:param action: electric of current input.
		:return:
		"""

		rad_last, rad_cur, rad_next, rad_next_next, error_last, error_cur = self.state

		error = trapezoidal_function(time=self.times[self.cnt]) - rad_cur

		if args.choose_model == "search_pid_parameter":
			kp, ki = action
			kp, ki = abs(kp), abs(ki)
			electric = self.pid.update_with_parameter(error=error, kp=kp, ki=ki, kd=0.0)
		else:
			electric = self.pid.update(error=error) + float(action)

		rad = self.update_rad(electric=electric)

		# Update the state.
		self.rad_last = self.rad_cur
		self.rad_cur = rad
		self.rad_next = trapezoidal_function(time = self.times[self.cnt + 1])
		self.rad_next_next = trapezoidal_function(time = self.times[self.cnt + 2])
		self.error_last = self.error_cur
		self.error_cur = error

		self.ec_last = electric
		self.cnt += 1

		if args.choose_model == "search_pid_parameter":
			reward = self._get_search_parameter_immediate_reward(error=error, electric=electric)
		else:
			reward = self._get_eletric_immediate_reward(error=error, electric=electric)

		done = self._get_termination()

		self.state = np.array([self.rad_last, self.rad_cur, self.rad_next, self.rad_next_next, self.error_last, self.error_cur])

		return self.state, reward, done

	def _get_eletric_immediate_reward(self, error, electric):
		"""
		Design the Reward Function of search immediate reward.
		:param error:
		:param electric:
		:return:
		"""

		error = min(error, 10000)

		mu = abs(error)

		sigma = 2

		reward = 0.9 * np.exp(-1 / 2 * (mu * mu) / (sigma * sigma)) + 0.1 * np.exp(-abs(electric))

		if np.isnan(reward): reward = 0  # error == inf -> nan -> reward == nan

		return reward

	def _get_search_parameter_immediate_reward(self, error, electric):
		"""
		Design the Reward Function of search PID parameter.
		:param error:
		:param electric:
		:return:
		"""

		error = min(error, 10000)

		mu = 0.9 * abs(error) + 0.1 * abs(electric)

		sigma = 10

		reward = np.exp(-1 / 2 * (mu * mu) / (sigma * sigma))

		if np.isnan(reward): reward = 0  # error == inf -> nan -> reward == nan

		return reward

	def _get_termination(self):
		"""
		Set the End Conditions.
		:return:
		"""

		done = abs(self.rad_cur) >= self.rad_threshold or abs(self.error_cur) >= args.height * 0.08

		return bool(done)

	def reset(self):
		"""
		#
		:return:
		"""

		self.rad_last = 0.0  # Speed at the previous moment
		self.rad_cur = 0.0  # Speed at the moment
		self.rad_next = 0.0  # Speed at the next moment
		self.rad_next_next = 0.0  # Speed at the next next moment
		self.error_last = 0.0  # The error at the previous moment
		self.error_cur = 0.0  # The error at the moment

		self.ec_last = 0.0  # electric current at the previous moment

		if args.choose_model == "search_pid_parameter":
			self.pid = PID(kp=args.kp, ki=args.ki, kd=args.kd, args=args)
		else:
			self.pid = PID(kp=args.kp_rl, ki=args.ki_rl, kd=args.kd_rl, args=args)

		self.state = np.array([self.rad_last, self.rad_cur, self.rad_next, self.rad_next_next, self.error_last, self.error_cur])
		self.cnt = 0
		return self.state

	def _system1(self, ec_cur):
		"""
		# system equation of system 1.
		:param ec_cur:
		:return:
		"""
		return self.rad_cur - 3.478 * 0.0001 * self.rad_last + 1.388 * ec_cur + 0.1986 * self.ec_last + 0.1 * np.random.normal(0, 1)

	def _system2(self, ec_cur):
		"""
		# system equation of system 2.
		:param ec_cur:
		:return:
		"""
		return self.rad_cur - 3.366 * 0.0001 * self.rad_last + 0.1263 * ec_cur + 0.01799 * self.ec_last + 0.1 * np.random.normal(0, 1)

	def _system3(self, ec_cur):
		"""
		# system equation of system 3.
		:param ec_cur:
		:return:
		"""
		return self.rad_cur - 3.478 * 0.0001 * self.rad_last + 1.388 * ec_cur + 0.1986 * self.ec_last + 0.1 * np.random.normal(0, 1) + 0.9148

if __name__ == "__main__":
	env = ServoSystemEnv()
	x = np.linspace(-10, 10, 100)
	y = [env._get_eletric_immediate_reward(error=i, electric=0) for i in x]
	import matplotlib.pyplot as plt
	plt.plot(x, y)
	plt.show()
