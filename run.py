# -*- coding: utf-8 -*-
# @Time    : 12/8/21 3:51 PM
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : run.py
# @Software: PyCharm


import os
import tensorflow as tf
import numpy as np
from parameter import args
from servo_system_env import ServoSystemEnv
from utils import trapezoidal_function

with tf.name_scope("S"):
	S = tf.placeholder(tf.float32, shape=[None, args.state_dim], name="s")

with tf.name_scope("R"):
	R = tf.placeholder(tf.float32, shape=[None, 1], name='r')

with tf.name_scope("S_"):
	S_ = tf.placeholder(tf.float32, shape=[None, args.state_dim], name="s_")


class Actor(object):
	def __init__(self, sess, action_dim, action_bound):
		self.sess = sess
		self.a_dim = action_dim
		self.a_bound = action_bound
		self.lr = args.learning_actor
		self.replace_iter = args.replace_iter_a
		self.replace_cnt = 0

		with tf.variable_scope("Actor"):
			# input s, output a
			self.a = self._build_net(S, scope="eval_net", trainable=True)

			# input s_, output a, get a_ for critic
			self.a_ = self._build_net(S, scope="target_net", trainable=True)

		self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor/eval_net")
		self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor/target_net")

		self.saver = tf.train.Saver()

	def _build_net(self, input, scope, trainable):

		with tf.variable_scope(scope):
			init_w = tf.contrib.layers.xavier_initializer()
			init_b = tf.constant_initializer(0.001)

			net1 = tf.layers.dense(inputs=input, units=200, activation=tf.nn.relu6, kernel_initializer=init_w,
								  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.reg),
								  bias_initializer=init_b, name="l1", trainable=trainable)

			net2 = tf.layers.dense(inputs=net1, units=200, activation=tf.nn.relu6, kernel_initializer=init_w,
								  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.reg),
								  bias_initializer=init_b, name="l2", trainable=trainable)

			net3 = tf.layers.dense(inputs=net2, units=10, activation=tf.nn.relu6, kernel_initializer=init_w,
								  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.reg),
								  bias_initializer=init_b, name="l3", trainable=trainable)

			with tf.variable_scope("a"):
				actions = tf.layers.dense(inputs=net3, units=self.a_dim, activation=tf.nn.tanh,
										  kernel_initializer=init_w, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.reg),
										  name="a", trainable=trainable)

				# Scale output to -action_bound to action_bound.
				scaled_a = tf.multiply(actions, self.a_bound, name="scaled_action")

			return scaled_a

	def learn(self, s):
		"""
		Batch Update
		:param s:
		:return:
		"""
		self.sess.run(self.train_op, feed_dict={S: s})

		if self.replace_cnt % self.replace_iter == 0:
			self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])

		self.replace_cnt += 1

	def choose_action(self, s):
		s = s[np.newaxis, :]
		return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

	def add_grad_to_graph(self, a_grads):
		with tf.variable_scope("policy_grads"):
			# ys = policy;
			# xs = policy's parameters;
			# a_grads = the gradients of the policy to get more Q;
			# tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da / dparams
			self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

		with tf.variable_scope("A_train"):
			opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
			self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
	def __init__(self, sess, a, a_, action_dim):
		self.sess = sess
		self.s_dim = args.state_dim
		self.a_dim = action_dim
		self.lr = args.learning_critic
		self.gamma = args.gamma

		self.replace_iter = args.replace_iter_c
		self.replace_cnt = 0

		with tf.variable_scope("Critic"):
			# Input (s, a), output q
			self.a = a
			self.q = self._build_net(S, self.a, "eval_net", trainable=True)

			# Input (s_, a_), output q_ for target
			self.q_ = self._build_net(S_, a_, "target_net", trainable=True)

			self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Critic/eval_net")
			self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Critic/target_net")

		with tf.variable_scope("target_q"):
			self.target_q = R + self.gamma * self.q_

		with tf.variable_scope("TD_error"):
			self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

		with tf.variable_scope("C_train"):
			self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

		with tf.variable_scope("a_grad"):
			self.a_grads = tf.gradients(self.q, self.a)[0]

		self.saver = tf.train.Saver()

	def _build_net(self, s, a, scope, trainable):
		with tf.variable_scope(scope):
			init_w = tf.contrib.layers.xavier_initializer()
			init_b = tf.constant_initializer(0.01)

			with tf.variable_scope('l1'):
				n_l1 = 200
				w1_s = tf.get_variable("w1_s", [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
				w1_a = tf.get_variable("w1_a", [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
				b1 = tf.get_variable("b1", [1, n_l1], initializer=init_b, trainable=trainable)
				net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

			net1 = tf.layers.dense(net, 200, activation=tf.nn.relu6, kernel_initializer=init_w,
								  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.reg),
								  bias_initializer=init_b, name="l2", trainable=trainable)

			net2 = tf.layers.dense(net1, 10, activation=tf.nn.relu6, kernel_initializer=init_w,
								  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.reg),
								  bias_initializer=init_b, name="l3", trainable=trainable)

			with tf.variable_scope("q"):
				q = tf.layers.dense(net2, 1, kernel_initializer=init_w,
									kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=args.reg),
									bias_initializer=init_b, trainable=trainable)

		return q

	def learn(self, s, a, r, s_):
		self.sess.run(self.train_op, feed_dict={S:s, self.a:a, R:r, S_:s_})

		if self.replace_cnt % self.replace_iter == 0:
			self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])

		self.replace_cnt += 1


class Memory(object):
	def __init__(self, capacity, dims):
		self.capacity = capacity
		self.data = np.zeros(shape=(capacity, dims))
		self.cnt = 0

	def store_transition(self, s, a, r, s_):
		transition = np.hstack((s, a, [r], s_))
		index = self.cnt % self.capacity
		self.data[index, :] = transition
		self.cnt += 1

	def sample(self, n):
		assert self.cnt >= self.capacity, print("Memory has not been fulfilled.")
		indices = np.random.choice(self.capacity, size=n)
		return self.data[indices, :]


class run_base(object):
	def __init__(self):
		current_dir = os.path.dirname(os.path.abspath(__file__))

		if args.run_type == "train":
			result_dir = os.path.join('results', 'ChooseModel_{}_CurveType_{}_Height_{}_DumpSystem_{}_RunType_{}'.format(args.choose_model, args.curve_type, args.height, args.use_jump_system, args.run_type))
			self.result_dir = os.path.join(current_dir, result_dir)
		else:
			# Need to load the train model.
			result_dir = os.path.join('results', 'ChooseModel_{}_CurveType_{}_Height_{}_DumpSystem_{}_RunType_{}'.format(args.choose_model, args.curve_type, args.height, args.use_jump_system, "train"))
			self.result_dir = os.path.join(current_dir, result_dir)

		os.makedirs(self.result_dir, exist_ok=True)

		if args.choose_model == "search_pid_parameter":
			self.max_episodes = 800
			self.action_dim = args.pid_parameter_search_action_dim
			self.action_bound = args.pid_parameter_search_action_bound
			self.var = 10

			tf.set_random_seed(args.seed)
			self.sess = tf.Session()

			self.actor = Actor(sess=self.sess, action_dim=self.action_dim, action_bound=self.action_bound)
			self.critic = Critic(sess=self.sess, a=self.actor.a, a_=self.actor.a_, action_dim=self.action_dim)

			self.actor.add_grad_to_graph(self.critic.a_grads)

			self.M = Memory(capacity=args.memory_capacity, dims=2 * args.state_dim + self.action_dim + 1)

			self.sess.run(tf.global_variables_initializer())

		elif args.choose_model == "search_electric":

			if args.curve_type == "trapezoidal":
				self.max_episodes = 800
			elif args.curve_type == "sine":
				self.max_episodes = 4000
			else:
				pass

			self.action_dim = args.electric_compensation_action_dim
			self.action_bound = args.electric_compensation_action_bound
			self.var = 2

			tf.set_random_seed(args.seed)
			self.sess = tf.Session()

			self.actor = Actor(sess=self.sess, action_dim=self.action_dim, action_bound=self.action_bound)
			self.critic = Critic(sess=self.sess, a=self.actor.a, a_=self.actor.a_, action_dim=self.action_dim)

			self.actor.add_grad_to_graph(self.critic.a_grads)

			self.M = Memory(capacity=args.memory_capacity, dims=2 * args.state_dim + self.action_dim + 1)

			self.sess.run(tf.global_variables_initializer())

		else:
			pass

		self.var_min = 0.01
		self.env = ServoSystemEnv()

	def train(self, times):
		pass

	def test(self, times):
		pass

	def load_model(self):

		self.actor.saver.restore(self.actor.sess, os.path.join(self.result_dir, "actor_model.ckpt"))
		self.critic.saver.restore(self.critic.sess, os.path.join(self.result_dir, "critic_model.ckpt"))

		print("load_model_successful.")

	def save_model(self):

		if not os.path.exists(self.result_dir):
			os.makedirs(self.result_dir)

		self.actor.saver.save(self.actor.sess, os.path.join(self.result_dir, "actor_model.ckpt"))
		self.critic.saver.save(self.critic.sess, os.path.join(self.result_dir, "critic_model.ckpt"))

		print("save_model_successful.")

	def save_result_txt(self, xs, ys, yaxis='radValues'):
		"""
		# Save the data to .txt file.
		:param xs:
		:param ys:
		:param yaxis:
		:return:
		"""

		filename = os.path.join(self.result_dir, yaxis + '.txt')
		if isinstance(xs, np.ndarray) and isinstance(ys, list):
			if not os.path.exists(filename):
				with open(file=filename, mode="a+") as file:
					file.write("times {}".format(yaxis))
					file.write("\r\n")
			else:
				print("{} has already existed. added will be doing.".format(filename))
				with open(file=filename, mode="a+") as file:
					file.write("times, {}".format(yaxis))
					file.write("\r\n")
		else:
			pass

		with open(file=filename, mode="a+") as file:
			for index, data in enumerate(zip(xs, ys)):
				file.write("{} {}".format(str(data[0]), str(data[1])))
				file.write("\r\n")


class run_class_pid(run_base):
	def __init__(self):
		super().__init__()

	def test(self, times):
		radValues = []
		iaeValues = []
		ecValues = []

		Episode_reward = 0

		for sim_time in times:

			error = trapezoidal_function(time=sim_time) - self.env.rad_cur

			ec_cur = self.env.ec_last

			state, reward, done = self.env.step(action=0)

			# append the value
			radValues.append(self.env.rad_cur)
			iaeValues.append(abs(error) if len(iaeValues) == 0 else abs(error) + iaeValues[-1])
			ecValues.append(ec_cur)

			Episode_reward += reward

		self.save_result_txt(xs=times, ys=radValues, yaxis="radValues")
		self.save_result_txt(xs=times, ys=iaeValues, yaxis="iaeValues")
		self.save_result_txt(xs=times, ys=ecValues, yaxis="ecValues")


class run_search_pid_parameter(run_base):
	def __init__(self):
		super().__init__()
		self.action_dim = args.pid_parameter_search_action_dim
		self.action_bound = args.pid_parameter_search_action_bound

	def train(self, times):
		steps = []
		epRewards = []
		maxEpReward = 4000.0  # set the base max reward.

		for i in range(self.max_episodes):
			s = self.env.reset()
			ep_reward = 0

			for j, sim_time in enumerate(times):

				# Add exploration noise
				a = self.actor.choose_action(s)
				a = np.clip(np.random.normal(a, self.var), -self.action_bound, self.action_bound)

				s_, r, done = self.env.step(a)

				self.M.store_transition(s, a, r, s_)

				if self.M.cnt > args.memory_capacity:
					self.var = max([self.var * 0.999995, self.var_min])  # decay the action randomness
					b_M = self.M.sample(args.batch_size)

					b_s = b_M[:, :args.state_dim]
					b_a = b_M[:, args.state_dim: args.state_dim + self.action_dim]
					b_r = b_M[:, - args.state_dim - 1: - args.state_dim]
					b_s_ = b_M[:, - args.state_dim:]

					self.critic.learn(b_s, b_a, b_r, b_s_)
					self.actor.learn(b_s)

				s = s_
				ep_reward += r

				if sim_time == times[-1]:
					print("Episode: {}, Reward {}, Explore {} Steps {}".format(i, ep_reward, self.var, j))
					epRewards.append(ep_reward)
					steps.append(i)

			if ep_reward >= maxEpReward:  # Save The Max Reward Model.
				print("Get More Episode Reward {}".format(maxEpReward))
				maxEpReward = ep_reward
				self.save_model()

		self.save_result_txt(xs=steps, ys=epRewards, yaxis="epRewards")


class run_search_compensation_electric(run_base):
	def __init__(self):
		super(run_search_compensation_electric, self).__init__()

	def train(self, times):
		steps = []
		epRewards = []
		maxEpReward = 4000.0  # set the base max reward.

		for i in range(self.max_episodes):
			s = self.env.reset()
			ep_reward = 0

			for j, sim_time in enumerate(times):

				# Add exploration noise
				a = self.actor.choose_action(s)
				a = np.clip(np.random.normal(a, self.var), -self.action_bound, self.action_bound)

				s_, r, done = self.env.step(a)

				self.M.store_transition(s, a, r, s_)

				if self.M.cnt > args.memory_capacity:
					self.var = max([self.var * 0.999995, self.var_min])  # decay the action randomness
					b_M = self.M.sample(args.batch_size)

					b_s = b_M[:, :args.state_dim]
					b_a = b_M[:, args.state_dim: args.state_dim + self.action_dim]
					b_r = b_M[:, - args.state_dim - 1: - args.state_dim]
					b_s_ = b_M[:, - args.state_dim:]

					self.critic.learn(b_s, b_a, b_r, b_s_)
					self.actor.learn(b_s)

				s = s_
				ep_reward += r

				if sim_time == times[-1]:
					print("Episode: {}, Reward {}, Explore {} Steps {}".format(i, ep_reward, self.var, j))
					epRewards.append(ep_reward)
					steps.append(i)

			if ep_reward >= maxEpReward:  # Save The Max Reward Model.
				print("Get More Episode Reward {}".format(maxEpReward))
				maxEpReward = ep_reward
				self.save_model()

		self.save_result_txt(xs=steps, ys=epRewards, yaxis="epRewards")

	def test(self, times):

		self.load_model()

		s = self.env.reset()
		ep_reward = 0

		radValues = []
		iaeValues = []
		ecValues = []

		for j, sim_time in enumerate(times):

			error = trapezoidal_function(time=sim_time) - self.env.rad_cur

			# Add exploration noise
			a = self.actor.choose_action(s)
			a = np.clip(np.random.normal(a, self.var), -self.action_bound, self.action_bound)

			ec_cur = self.env.ec_last

			s_, r, done = self.env.step(a)

			# append the value
			radValues.append(self.env.rad_cur)
			iaeValues.append(abs(error) if len(iaeValues) == 0 else abs(error) + iaeValues[-1])
			ecValues.append(ec_cur)

			s = s_
			ep_reward += r

		self.save_result_txt(xs=times, ys=radValues, yaxis="radValues")
		self.save_result_txt(xs=times, ys=iaeValues, yaxis="iaeValues")
		self.save_result_txt(xs=times, ys=ecValues, yaxis="ecValues")

def main():
	times = np.arange(args.start_time, args.end_time, args.dt)[:args.max_ep_steps]

	if args.run_type == "train":
		if args.choose_model == "search_pid_parameter":
			run_search_pid_parameter().train(times=times)
		elif args.choose_model == "search_electric":
			run_search_compensation_electric().train(times=times)
		elif args.choose_model == "class_pid":
			run_class_pid().test(times=times)
		else:
			print("Please Input The Right Choose Model.")
	else:
		if args.choose_model == "search_pid_parameter":
			run_search_pid_parameter().test(times=times)
		elif args.choose_model == "search_electric":
			run_search_compensation_electric().test(times=times)
		elif args.choose_model == "class_pid":
			run_class_pid().test(times=times)
		else:
			print("Please Input The Right Choose Model.")


if __name__ == "__main__":
	main()