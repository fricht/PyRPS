import numpy as np
import random


class Network:
	activation = np.tanh
	# TODO : make sure tanh is handled correctely

	def __init__(self, params):
		if type(params[0]) is int:
			self.generate(*params)
		else:
			self.params = params

	def generate(self, input_size, hidden, output_size):
		params = []
		for layer, inputs in zip(hidden + [output_size], [input_size] + hidden):
			params.append(np.random.normal(scale=2.0, size=(layer, inputs + 1)))
		self.params = tuple(params)

	def feed_forward(self, inputs):
		inputs = np.append(inputs, 1.0)
		outputs = None
		for layer in self.params:
			outputs = np.empty(shape=(layer.shape[0]))
			for n, cell in enumerate(layer):
				outputs[n] = self.activation(np.dot(inputs, cell))
			inputs = np.append(outputs, 1.0)
		return outputs
