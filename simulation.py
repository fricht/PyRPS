from typing import Any
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

	def step(self: 'Entity', vision: np.ndarray) -> np.ndarray:
		net_response: np.ndarray = self.network.feed_forward(np.append(vision.flatten(), np.array([self.age, self.energy])))
		self.age += 1
		self.energy -= self.loss * (self.age / 100) ** 2
		self._signal = net_response[3]
		net_response[0] = net_response[0] * 2 - 1
		net_response[1] = net_response[1] * 2 - 1
		return net_response

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

	def __init__(self: 'Simulation', grid_size: tuple[int, int], pop_size: int, internal_neurons: list[int], data: dict[str, Any]) -> None:
		self.speed = data['speed']
		self.damage = data['damage']
		self.steal = data['steal']
		self.range = data['range']
		self.energy = data['energy']
		self.loss_factor = data['loss_factor']
		self.vision = data['vision']
		self.grid_size: tuple[int, int] = grid_size
		self.map: np.ndarray = np.empty(shape=self.grid_size, dtype=object)
		self.tick: int = 0
		self.generate(pop_size, internal_neurons)

	def generate(self: 'Simulation', pop_size: int, net_size: list[int]) -> None:
		for _ in range(pop_size):
			# Rock
			self.map[
				random.randint(0, self.grid_size[0] - 1),
				random.randint(0, self.grid_size[0] - 1)
			] = Entity(0, Network([(2 * self.vision[0] + 1) ** 2 * 3 - 1, net_size, 4]), self.energy[0][0], self.loss_factor[0])
			# Paper
			self.map[
				random.randint(0, self.grid_size[0] - 1),
				random.randint(0, self.grid_size[0] - 1)
			] = Entity(1, Network([(2 * self.vision[1] + 1) ** 2 * 3 - 1, net_size, 4]), self.energy[1][0], self.loss_factor[1])
			# Scissors
			self.map[
				random.randint(0, self.grid_size[0] - 1),
				random.randint(0, self.grid_size[0] - 1)
			] = Entity(2, Network([(2 * self.vision[0] + 1) ** 2 * 3 - 1, net_size, 4]), self.energy[2][0], self.loss_factor[2])

	def delta_entity_type(self: 'Simulation', e_ref: int, e: int) -> int:
		if e_ref == e:
			return 1
		if abs(e_ref - e) == 1:
			return 4 * int(e_ref < e) - 2
		else:
			return 4 * int(e_ref > e) - 2

	def step(self: 'Simulation') -> None:
		self.tick += 1
		# TODO : change the way to handle movements, for now, entities can destroy others by just 'overwriting' them
		new_map: np.ndarray = np.empty(shape=self.grid_size, dtype=object)
		for ind, entity in np.ndenumerate(self.map):
			if entity is not None:
				if entity.energy <= 0:
					continue
				# compute vision
				vision: np.ndarray = np.zeros(shape=((2 * self.vision[entity.type] + 1) ** 2 - 1, 3))
				t: int = -1  # tracker for vision index (simpler)
				for dx in range(-self.vision[entity.type], self.vision[entity.type] + 1):
					for dy in range(-self.vision[entity.type], self.vision[entity.type] + 1):
						if dx == dy == 0:
							continue
						t += 1
						x: int = ind[0] + dx
						y: int = ind[1] + dy
						if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):
							vision[t, 0] = -1
							continue
						if self.map[x, y] is None:
							continue
						d_e_type: int = self.delta_entity_type(entity.type, self.map[x, y].type)
						vision[t, 0] = d_e_type
						vision[t, 1] = self.map[x, y].energy
						vision[t, 2] = self.map[x, y].signal * int(d_e_type == 1)
				# process NN
				action = entity.step(vision)
				# apply movements
				new_pos = (
					min(max(round(ind[0] + action[0] * self.speed[entity.type]), 0), self.grid_size[0] - 1),
					min(max(round(ind[1] + action[1] * self.speed[entity.type]), 0), self.grid_size[1] - 1)
				)
				new_map[new_pos] = entity
				# TODO : handle new born
				# TODO : handle killing
		for _, entity in np.ndenumerate(new_map):  # , flags=['refs_ok']
			if entity is not None:
				entity.sub_process()
		self.map = new_map
