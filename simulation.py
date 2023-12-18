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

	def damage(self: 'Entity', amount: float) -> float:
		delta: float = min(abs(self.energy), amount)
		self.energy -= amount
		return delta

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
	# logs
	log_0: list[int] = []
	log_1: list[int] = []
	log_2: list[int] = []
	log_t: list[int] = []
	# hyperparameters
	mutation_rate: float = 0.02
	change_rate: float = 0.002
	# Specie characteristics
	speed: tuple[int, int, int]  # square or circle ?
	damage: tuple[float, float, float]
	steal: tuple[float, float, float]
	energy: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]  # born, need to reproduce
	loss_factor: tuple[float, float, float]  # evergy loss over time : ax^2, where 'a' is the loss factor
	vision: tuple[int, int, int]  # square or circle ?

	def __init__(self: 'Simulation', grid_size: tuple[int, int], pop_size: int, internal_neurons: list[int], data: dict[str, Any]) -> None:
		self.speed = data['speed']
		self.damage = data['damage']
		self.steal = data['steal']
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
			] = Entity(2, Network([(2 * self.vision[2] + 1) ** 2 * 3 - 1, net_size, 4]), self.energy[2][0], self.loss_factor[2])

	def delta_entity_type(self: 'Simulation', e_ref: int, e: int) -> int:
		if e_ref == e:
			return 1
		if abs(e_ref - e) == 1:
			return 4 * int(e_ref < e) - 2
		else:
			return 4 * int(e_ref > e) - 2

	def step(self: 'Simulation') -> bool:
		self.log_0.append(0)
		self.log_1.append(0)
		self.log_2.append(0)
		self.log_t.append(0)
		self.tick += 1
		# (TODO) : change the way to handle movements, for now, entities can destroy others by just 'overwriting' them
		# collisions handled, maybe it's a solution
		new_map: np.ndarray = np.empty(shape=self.grid_size, dtype=object)
		for ind, entity in np.ndenumerate(self.map):
			if entity is not None:
				if entity.energy <= 0:
					continue
				# compute vision
				vision: np.ndarray = np.zeros(shape=((2 * self.vision[entity.type] + 1) ** 2 - 1, 3))
				food: list[list[object, int]] = []
				t: int = -1  # tracker for vision index (simpler)
				for dx in range(-self.vision[entity.type], self.vision[entity.type] + 1):
					for dy in range(-self.vision[entity.type], self.vision[entity.type] + 1):
						if dx == dy == 0:
							continue
						t += 1
						x: int = (ind[0] + dx) % self.grid_size[0]
						y: int = (ind[1] + dy) % self.grid_size[1]
						if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):  # should never be True
							print('WTF ?!?')
							vision[t, 0] = -1
							continue
						elem: None | Entity = self.map[x, y]
						if elem is None:
							continue
						d_e_type: int = self.delta_entity_type(entity.type, elem.type)
						if d_e_type == 2 and abs(dx) <= self.range[entity.type] and abs(dy) <= self.range[entity.type]:
							food.append([elem, x**2 + y**2])
						vision[t, 0] = d_e_type
						vision[t, 1] = elem.energy
						vision[t, 2] = elem.signal * int(d_e_type == 1)
				# process NN
				action = entity.step(vision)
				new_pos: tuple[int, int] = (  # used later
					round(ind[0] + action[0] * self.speed[entity.type]) % self.grid_size[0],
					round(ind[1] + action[1] * self.speed[entity.type]) % self.grid_size[1]
				)
				# handle newborn
				if entity.energy >= self.energy[entity.type][1]:
					entity.energy /= 2
					child: Entity = Entity(
						entity.type,
						entity.network.child(self.mutation_rate, self.change_rate),
						self.energy[entity.type][0],
						self.loss_factor[entity.type]
					)
					if new_map[new_pos] is None:
						new_map[new_pos] = child
						self.log_t[-1] += 1
						self.log_0[-1] += int(child.type == 0)
						self.log_1[-1] += int(child.type == 1)
						self.log_2[-1] += int(child.type == 2)
					else:
						possibles_pos: list[list[int, int, int]] = []
						for dx in range(-self.vision[entity.type], self.vision[entity.type] + 1):
							for dy in range(-self.vision[entity.type], self.vision[entity.type] + 1):
								x: int = (new_pos[0] + dx) % self.grid_size[0]
								y: int = (new_pos[1] + dy) % self.grid_size[1]
								if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):  # should never be True
									continue
								if new_map[x, y] is None:
									possibles_pos.append([x, y, dx ** 2 + dy ** 2])
						if len(possibles_pos) > 0:
							possibles_pos.sort(key=lambda a: a[2])
							new_map[possibles_pos[0][0], possibles_pos[0][1]] = child
							self.log_t[-1] += 1
							self.log_0[-1] += int(child.type == 0)
							self.log_1[-1] += int(child.type == 1)
							self.log_2[-1] += int(child.type == 2)
						else:
							entity.energy *= 2  # gives back energy
				# apply movements
				if new_map[new_pos] is None:
					new_map[new_pos] = entity
					self.log_t[-1] += 1
					self.log_0[-1] += int(entity.type == 0)
					self.log_1[-1] += int(entity.type == 1)
					self.log_2[-1] += int(entity.type == 2)
				else:
					possibles_pos: list[list[int, int, int]] = []
					for dx in range(-self.vision[entity.type], self.vision[entity.type] + 1):
						for dy in range(-self.vision[entity.type], self.vision[entity.type] + 1):
							x: int = (new_pos[0] + dx) % self.grid_size[0]
							y: int = (new_pos[1] + dy) % self.grid_size[1]
							if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):  # should never be True
								continue
							if new_map[x, y] is None:
								possibles_pos.append([x, y, dx**2 + dy**2])
					if len(possibles_pos) > 0:
						possibles_pos.sort(key=lambda a: a[2])
						new_map[possibles_pos[0][0], possibles_pos[0][1]] = entity
						self.log_t[-1] += 1
						self.log_0[-1] += int(entity.type == 0)
						self.log_1[-1] += int(entity.type == 1)
						self.log_2[-1] += int(entity.type == 2)
				# handle killing
				if action[2] > 0.6 and len(food) > 0:
					food.sort(key=lambda a: a[1])
					entity.energy += food[0][0].damage(self.damage[entity.type]) * self.steal[entity.type]
		for _, entity in np.ndenumerate(new_map):
			if entity is not None:
				entity.sub_process()
		self.map = new_map
		return self.log_t[-1] > 0
