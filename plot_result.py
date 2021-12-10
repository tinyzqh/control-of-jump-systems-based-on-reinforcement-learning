# -*- coding: utf-8 -*-
# @Time    : 12/8/21 7:47 PM
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : plot_result.py
# @Software: PyCharm

import os
import pandas as pd
import seaborn as sns
from parameter import args
import matplotlib.pyplot as plt

class plot_result(object):
	def __init__(self, root_path):
		"""
		:param root_path:
		"""
		self.root_path = root_path
		if self.root_path.split('_')[1] == "class":
			self.label = "Classic PID"
		else:
			self.label = "RL PID"

	def get_files_path(self):
		"""
		# Get All '.txt' files.
		:return:
		"""
		file_path_dir = []
		for root, dir, files in os.walk(self.root_path):
			for file in files:
				file_path = os.path.join(root, file)
				file_path_dir.append(file_path)
		return file_path_dir

	def get_data(self):
		files = self.get_files_path()

		DataDict = {}
		for file in files:
			if file.split('/')[-1].split('.')[-1] == "txt":
				print("filename {} processed.".format(file))
				DictName = file.split('/')[-1].split('.')[0]
				df = pd.read_csv(file, sep=" ")
				ColumnsName = df.columns
				Dict = {}
				Dict[ColumnsName[0]] = df[ColumnsName[0]]
				Dict[ColumnsName[1]] = df[ColumnsName[1]]

				DataDict[DictName] = Dict
			else:
				print("filename {} not .txt file.".format(file))

		return DataDict

	def plot_data(self):
		"""
		plot the all .txt files.
		:return:
		"""
		sns.set(style="darkgrid", font_scale=1.5, font='serif', rc={'figure.figsize': (10, 8)})
		DataDict = self.get_data()
		for key, data in DataDict.items():

			plt.plot(data['times'], data[key], label=self.label)
			plt.xlabel("Time (s)")
			plt.ylabel(key)
			plt.legend()
			plt.savefig(self.root_path + '/{}.png'.format(key))
			plt.show()


if __name__ == "__main__":
	current_dir = os.path.dirname(os.path.abspath(__file__))
	result_dir = os.path.join('results', 'ChooseModel_{}_CurveType_{}_Height_{}_DumpSystem_{}_RunType_{}'.format(args.choose_model, args.curve_type, args.height, args.use_jump_system, args.run_type))

	root_path = os.path.join(current_dir, result_dir)
	file_path_dir = plot_result(root_path=root_path).plot_data()