# -*- coding: utf-8 -*-
# @Time    : 11/29/21 9:47 PM
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : servo_system.py
# @Software: PyCharm

from parameter import args
import numpy as np
import gym

class ServoSystem(gym.Env):
	def __init__(self):
		super(ServoSystem, self).__init__()
		self.rad_cur = 0.0  # Speed at the moment
		self.rad_last = 0.0  # Speed at the previous moment
		self.ec_last = 0.0  # electric current at the previous moment
		self.cnt = 0  # used to count the step number

		self.times = np.arange(args.start_time, args.end_time, args.dt)  # generate the time steps.
		np.random.seed(args.seed)
		self.SystemsList = np.random.randint(0, 3, args.change_times)  # Generate the random systems id.
		self.SystemsLists = [i for i in self.SystemsList for j in range(int(len(self.times)/args.change_times) + 1)] # extended system id.

	def step(self, action):
		"""
		:param action: electric of current input.
		:return:
		"""
		if args.use_jump_system:
			if self.SystemsLists[self.cnt] == 0:
				rad = self._system1(ec_cur = action)
			elif self.SystemsLists[self.cnt] == 1:
				rad = self._system2(ec_cur = action)
			else:
				assert self.SystemsLists[self.cnt] == 2, print("the system id of SystemsLists is out of range.")
				rad = self._system3(ec_cur = action)
		else:
			if args.used_system_id == 1:
				rad = self._system1(ec_cur = action)
			elif args.used_system_id == 2:
				rad = self._system2(ec_cur = action)
			else:
				assert args.used_system_id == 3, print("the selected system id is illegal")
				rad = self._system2(ec_cur=action)

		self.rad_last = self.rad_cur
		self.rad_cur = rad
		self.ec_last = action

		self.cnt += 1

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

	def reset(self):
		self.cnt = 0

if __name__ == "__main__":
	pass

