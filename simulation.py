from typing import Any
import numpy as np
import random


class Network:
	"""
	Neural Network (NN) class.
	Used for the simulation.
	"""
	def __init__(self: 'Network', params: list[int, list[int], int] | tuple[np.ndarray]) -> None:
		"""
		Initializes the Network Class

		Parameters:
			params: tuple of numpy arrays for the weights and biases (from another Network)
				or list : [int: inputs, [int, ...]: hidden, int: output] to create a new random Network
		"""
		if type(params) is list:
			self.generate(params[0], params[1], params[2])
		else:
			self.params = params

	def generate(self: 'Network', input_size: int, hidden: list[int], output_size: int) -> None:
		"""
		Initialize the network.

		Parameters:
			input_size: an integer representing the number of inputs of the Network
			hidden: a list of int representing the hidden neurons, layer by layer
			output_size: an integer representing the number of outputs of the Network

		Returns:
			None
		"""
		params: list[np.ndarray] = []
		for layer, inputs in zip(hidden + [output_size], [input_size] + hidden):
			params.append(np.random.normal(scale=2.0, size=(layer, inputs + 1)))
		self.params: tuple[np.ndarray] = tuple(params)

	def activation(self: 'Network', input_value: float) -> float:
		"""
		The activation function for the Network.

		Parameters:
			input_value: a float at which the activation function will be applied

		Returns:
			float, the result of the activation function
		"""
		return 1 / (1 + np.exp(- input_value))

	def feed_forward(self: 'Network', inputs: np.ndarray) -> np.ndarray:
		"""
		Feed Forward the Network

		Parameters:
			inputs: a numpy array, with size matching to the number given to initialize the Network.

		Returns:
			a numpy array, the output of the Network.
		"""
		inputs = np.append(inputs, 1.0)
		output: np.ndarray
		for layer in self.params:
			output = np.empty(shape=(layer.shape[0]))
			for n, cell in enumerate(layer):
				output[n] = self.activation(np.dot(inputs, cell))
			inputs = np.append(output, 1.0)
		return output

	def child(self: 'Network', mut_rate: float, mod_rate: float) -> 'Network':
		"""
		A method to create a child of the Network.

		Parameters:
			mut_rate: a float in [0, 1] representing the probability of a mutation in the child Network.
			mod_rate: a float in [0, 1] representing the probability of a modification in the child Network.

		Returns:
			a new instance of the Network class.
		"""
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
	"""
	A class representing a single entity.
	"""
	def __init__(self: 'Entity', e_type: int, network: Network, energy: float, loss: float) -> None:
		"""
		Initializes the Entity class.

		Parameters:
			e_type: int in [0, 2] representing the type of the entity
			network: the Network object for the 'brain' of the Entity.
			energy: float, the initial amount of energy
			loss: float, the rate at which the energy is lost over time ( loss * (age / 100) ^ 2 )
		"""
		self.type = e_type
		self.network: Network = network
		self.energy: float = energy
		self.loss: float = loss
		self.signal: float = 0.0
		self._signal: float = 0.0
		self.age: int = 0

	def damage(self: 'Entity', amount: float) -> float:
		"""
		Apply damage to the Entity and returns the amount of damage effectively applied.

		Parameters:
			amount: float, the maximum amount of damage that can be applied

		Returns:
			float, the amount of damage applied
		"""
		delta: float = min(abs(self.energy), amount)
		self.energy -= amount
		return delta

	def step(self: 'Entity', vision: np.ndarray) -> np.ndarray:
		"""
		Apply a single step for the simulation for this Entity.

		Parameters:
			vision: numpy array, what this entity sees

		Returns:
			numpy array, the actions taken by the entity
		"""
		net_response: np.ndarray = self.network.feed_forward(np.append(vision.flatten(), np.array([self.age, self.energy])))
		self.age += 1
		self.energy -= self.loss * (self.age / 100) ** 2
		self._signal = net_response[3]
		net_response[0] = net_response[0] * 2 - 1
		net_response[1] = net_response[1] * 2 - 1
		return net_response

	def sub_process(self: 'Entity') -> None:
		"""
		Apply subprocess actions.

		Returns:
			None
		"""
		self.signal = self._signal


class Simulation:
	"""
	A class representing the Simulation.

	To change the mutation_rate or the change_rate, change the attributes of the instantiated class.
	"""
	# logs
	log_0: list[int] = []
	log_1: list[int] = []
	log_2: list[int] = []
	log_t: list[int] = []
	# hyperparameters
	mutation_rate: float = 0.02
	change_rate: float = 0.002
	# Specie characteristics
	speed: tuple[int, int, int]
	damage: tuple[float, float, float]
	steal: tuple[float, float, float]
	energy: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]  # born, need to reproduce
	loss_factor: tuple[float, float, float]  # evergy loss over time : ax^2, where 'a' is the loss factor
	vision: tuple[int, int, int]
	range: tuple[int, int, int]

	def __init__(self: 'Simulation', grid_size: tuple[int, int], pop_size: int, internal_neurons: list[int], data: dict[str, Any]) -> None:
		"""
		Initializes the Entity class.

		Parameters:
			grid_size: a tuple [int, int] for the dimensions of the simulation
			pop_size: an integer, for the amount of each specie initially on the board
			internal_neurons: a list of integer, representing the hidden neurons of each Entity.
			data: a dictionary with the characteristics of each specie.
				ex : data = {  # each value is in the form (specie0, specie1, specie2):
							'speed': (5, 5, 5),  # int, the max distance an Entity can travel each step
							'damage': (8, 8, 8),  # float, the amount of damage an entity deals
							'steal': (0.7, 0.7, 0.7),  # float, the proportion of damage dealt that the Entity gain
							'energy': ((70, 100), (70, 100), (70, 100)),  # (float, float), the initial energy of a specie, the amount required to create a child
							'loss_factor': (0.07, 0.07, 0.07),  # float, the rate of natural decrease of the energy of an Entity
							'vision': (12, 12, 12),  # int, the vision of an Entity
							'range': (5, 5, 5),  # int, the attack range, must be lower than or equals to the vision
							}
		"""
		self.speed = data['speed']
		self.damage = data['damage']
		self.steal = data['steal']
		self.energy = data['energy']
		self.loss_factor = data['loss_factor']
		self.vision = data['vision']
		self.range = data['range']
		self.grid_size: tuple[int, int] = grid_size
		self.map: np.ndarray = np.empty(shape=self.grid_size, dtype=object)
		self.tick: int = 0
		self.generate(pop_size, internal_neurons)

	def generate(self: 'Simulation', pop_size: int, net_size: list[int]) -> None:
		"""
		Generates the initial population.

		Parameters:
			pop_size: int, the initial amount of Entity of each Specie
			net_size: list of int, the hidden neurons

		Returns:
			None
		"""
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
		"""
		Get the relation between 2 Entities (friends / prey / predator).

		Parameters:
			e_ref: int in [0, 2], the type of the reference entity
			e: int in [0, 2], the type of the other entity

		Returns:
			int, the relation of the reference entity on the other one.
				1 if friends
				2 if e is a prey for e_ref
				-2 if e is a predator of e_ref
		"""
		if e_ref == e:
			return 1
		if abs(e_ref - e) == 1:
			return 4 * int(e_ref < e) - 2
		else:
			return 4 * int(e_ref > e) - 2

	def step(self: 'Simulation') -> bool:
		"""
		Performs one step in the Simulation.

		Returns:
			bool, True if the simulation continues, False if everyone is dead
		"""
		self.log_0.append(0)
		self.log_1.append(0)
		self.log_2.append(0)
		self.log_t.append(0)
		self.tick += 1
		# change the way to handle movements, for now, entities can destroy others by just 'overwriting' them
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
