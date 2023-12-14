import random
import numpy as np
import random


class Network:
	def __init__(self, params: list[int, list[int], int] | tuple[np.ndarray]) -> None:
		if type(params) is list:
			self.generate(params[0], params[1], params[2])
		else:
			self.params = params

	def generate(self: 'Network', input_size: int, hidden: list[int], output_size: int) -> None:
		params: list[np.ndarray] = []
		for layer, inputs in zip(hidden + [output_size], [input_size] + hidden):
			params.append(np.random.normal(scale=2.0, size=(layer, inputs + 1)))
		self.params: tuple[np.ndarray] = tuple(params)

	def activation(self: 'Network', input_value: float) -> float:
		return 1 / (1 + np.exp(- input_value))

	def feed_forward(self: 'Network', inputs: np.ndarray) -> np.ndarray:
		inputs = np.append(inputs, 1.0)
		output: np.ndarray
		for layer in self.params:
			output = np.empty(shape=(layer.shape[0]))
			for n, cell in enumerate(layer):
				output[n] = self.activation(np.dot(inputs, cell))
			inputs = np.append(output, 1.0)
		return output

	def child(self: 'Network', mut_rate, mod_rate) -> 'Network':
		params: list[np.ndarray] = [arr.copy() for arr in self.params]
		for layer in params:
			for ind, val in np.ndenumerate(layer):
				n: float = random.random()
				if n < mod_rate:
					layer[ind] = random.uniform(-4, 4)
				elif n < mut_rate:
					layer[ind] = val + random.random()
		return Network(tuple(params))


class Entity:
	def __init__(self: 'Entity', e_type: int, network: Network, energy: float, loss: float) -> None:
		self.type = e_type
		self.network: Network = network
		self.energy: float = energy
		self.loss: float = loss
		self.signal: float = 0.0
		self._signal: float = 0.0
		self.age: int = 0

	def step(self: 'Entity', vision: np.ndarray) -> None:
		net_response: np.ndarray = self.network.feed_forward(vision.flatten())
		self.age += 1
		self.energy -= self.loss * (self.age / 100) ** 2
		self._signal = net_response[3]

	def sub_process(self: 'Entity') -> None:
		self.signal = self._signal


class Simulation:
	# hyperparameters
	mutation_rate: float = 0.01
	change_rate: float = 0.001
	# Specie characteristics
	speed: tuple[int, int, int]  # square or circle ?
	damage: tuple[float, float, float]
	steal: tuple[float, float, float]
	range: tuple[int, int, int]  # range to kill entities
	energy: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]  # born, need to reproduce
	loss_factor: tuple[float, float, float]  # evergy loss over time : ax^2, where 'a' is the loss factor
	vision: tuple[int, int, int]  # square or circle ?

	def __init__(self: 'Simulation', grid_size: tuple[int, int], pop_size: int, internal_neurons: list[int]) -> None:
		self.grid_size: tuple[int, int] = grid_size
		self.map: np.ndarray = np.empty(shape=self.grid_size, dtype=object)
		self.tick: int = 0
		self.generate(pop_size, grid_size, internal_neurons)

	def generate(self: 'Simulation', pop_size: int, grid_size: tuple[int, int], net_size: list[int]) -> None:
		for _ in range(pop_size):
			# Rock
			self.map[
				random.randint(0, grid_size[0] - 1),
				random.randint(0, grid_size[0] - 1)
			] = Entity(0, Network(net_size))
			# Paper
			self.map[
				random.randint(0, grid_size[0] - 1),
				random.randint(0, grid_size[0] - 1)
			] = Entity(1, Network(net_size))
			# Scissors
			self.map[
				random.randint(0, grid_size[0] - 1),
				random.randint(0, grid_size[0] - 1)
			] = Entity(2, Network(net_size))

	def step(self: 'Simulation') -> None:
		# TODO : change the way to handle movements, for now, entities can destroy others by just 'walking' on them
		new_map: np.ndarray = np.empty(shape=self.grid_size, dtype=object)
		for ind, entity in np.ndenumerate(self.map):
			pass
		self.map = new_map
