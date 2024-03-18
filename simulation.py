import numpy as np
import random

class Network:
    def new(params):
        net = Network()
        net.generate(*params)
        return net

    def from_params(params):
        net = Network()
        net.params = params
        return net

    def generate(self, input_size, hidden, output_size):
        params = []
        for layer, inputs in zip(hidden + [output_size], [input_size] + hidden):
            params.append(np.random.normal(scale=2.0, size=(layer, inputs + 1)))
        self.params = params

    def activation(self, input_value):
        if abs(input_value) > 100:
            return 1 - int(input_value < 0)
        return 1 / (1 + np.exp(- input_value))

    def feed_forward(self, inputs):
        inputs = np.append(inputs, 1.0)
        output = None
        for layer in self.params:
            output = np.empty(shape=(layer.shape[0]))
            for n, cell in enumerate(layer):
                output[n] = self.activation(np.dot(inputs, cell))
            inputs = np.append(output, 1.0)
        return output

    def child(self, mod_scale):
        params = []
        for layer in self.params:
            params.append(np.add(layer, np.random.normal(scale=mod_scale, size=layer.shape)))
        return Network.from_params(params)


# why the fuck is numpy slower than python iterations ???
# class _Network:
# 	def __init__(self, params, gen=False):
# 		if gen:
# 			self.generate(params[0], params[1], params[2])
# 		else:
# 			self.params = params

# 	def generate(self, input_size, hidden, output_size):
# 		params = []
# 		for layer, inputs in zip(hidden + [output_size], [input_size] + hidden):
# 			params.append(np.random.normal(scale=2.0, size=(inputs + 1, layer)))
# 		self.params = params

# 	@np.vectorize
# 	def activation(input_value):
# 		return 1 / (1 + np.exp(- input_value))

# 	def feed_forward(self, inputs):
# 		inputs = np.array([np.append(inputs, 1.0)])
# 		output = None
# 		for layer in self.params:
# 			output = Network.activation(np.matmul(inputs, layer))
# 			inputs = np.append(output, 1.0)
# 		return output

# 	def child(self, mut_rate, mod_rate):
# 		params = []
# 		for layer in self.params:
# 			params.append(np.add(layer, np.random.normal(scale=mod_rate, size=layer.shape)))
# 		return Network(params)


class Entity:
    class Types:
        PAPER = 0
        ROCK = 1
        SCISSORS = 2

    def __init__(self, entity_type, network, energy, loss):
        self.type = entity_type
        self.network = network
        self.energy = energy
        self.loss = loss
        self.signal = 0.0
        self._signal = 0.0
        self.age = 0

    def damage(self, amount):
        delta = min(abs(self.energy), amount)
        self.energy -= amount
        return delta

    def step(self, vision):
        net_response = self.network.feed_forward(np.append(vision.flatten(), np.array([self.age, self.energy])))
        self.age += 1
        self.energy -= self.loss * (self.age / 100) ** 2
        self._signal = net_response[3]
        net_response[0] = net_response[0] * 2 - 1
        net_response[1] = net_response[1] * 2 - 1
        return net_response

    def sub_process(self):
        self.signal = self._signal


class Simulation:
    def __init__(self, grid_size, pop_size, internal_neurons, data):
        # logs
        self.log_0 = []
        self.log_1 = []
        self.log_2 = []
        self.log_t = []
        # Specie characteristics
        print(data) # ???????????????????? why aren't they like i gave them ?
        self.mod_scale = data['mod_scale']
        self.speed = data['speed']
        self.damage = data['damage']
        self.steal = data['steal']
        self.energy = data['energy']  # born, need to reproduce
        self.loss_factor = data['loss_factor'] # energy loss over time : ax^2, where 'a' is the loss factor
        self.vision = data['vision']
        self.range = data['range']
        self.grid_size = grid_size
        self.pop_size = pop_size
        self.internal_neurons = internal_neurons
        self.map = np.empty(shape=self.grid_size, dtype=object)
        self.tick = 0
        self.generate(pop_size, internal_neurons)

    def reset(self): # should be useless
        self.map = np.empty(shape=self.grid_size, dtype=object)
        self.tick = 0
        self.log_0 = []
        self.log_1 = []
        self.log_2 = []
        self.log_t = []
        self.generate(self.pop_size, self.internal_neurons)

    def generate(self, pop_size, net_size):
        for _ in range(pop_size):
            rock = Entity(Entity.Types.ROCK, Network.new([(2 * self.vision[0] + 1) ** 2 * 3 - 1, net_size, 4]), self.energy[0][0], self.loss_factor[0])
            paper = Entity(Entity.Types.PAPER, Network.new([(2 * self.vision[1] + 1) ** 2 * 3 - 1, net_size, 4]), self.energy[1][0], self.loss_factor[1])
            scissors = Entity(Entity.Types.SCISSORS, Network.new([(2 * self.vision[2] + 1) ** 2 * 3 - 1, net_size, 4]), self.energy[2][0], self.loss_factor[2])
            self.place_entity_random(rock)
            self.place_entity_random(paper)
            self.place_entity_random(scissors)

    def place_entity_random(self, entity):
        x = random.randint(0, self.grid_size[0] - 1)
        y = random.randint(0, self.grid_size[0] - 1)
        self.map[x, y] = entity

    def delta_entity_type(self, e_ref, e):
        if e_ref == e:
            return 1
        if abs(e_ref - e) == 1:
            return 4 * int(e_ref < e) - 2
        else:
            return 4 * int(e_ref > e) - 2

    def step(self):
        self.log_0.append(0)
        self.log_1.append(0)
        self.log_2.append(0)
        self.log_t.append(0)
        self.tick += 1
        # change the way to handle movements, for now, entities can destroy others by just 'overwriting' them
        # collisions handled, maybe it's a solution
        new_map = np.empty(shape=self.grid_size, dtype=object)
        for ind, entity in np.ndenumerate(self.map):
            if entity is not None:
                if entity.energy <= 0:
                    continue
                # compute vision
                vision = np.zeros(shape=((2 * self.vision[entity.type] + 1) ** 2 - 1, 3))
                food = []
                t = -1  # tracker for vision index (simpler)
                for dx in range(-self.vision[entity.type], self.vision[entity.type] + 1):
                    for dy in range(-self.vision[entity.type], self.vision[entity.type] + 1):
                        if dx == dy == 0:
                            continue
                        t += 1
                        x = (ind[0] + dx) % self.grid_size[0]
                        y = (ind[1] + dy) % self.grid_size[1]
                        if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):  # should never be True
                            print('WTF ?!?')
                            vision[t, 0] = -1
                            continue
                        elem = self.map[x, y]
                        if elem is None:
                            continue
                        d_e_type = self.delta_entity_type(entity.type, elem.type)
                        if d_e_type == 2 and abs(dx) <= self.range[entity.type] and abs(dy) <= self.range[entity.type]:
                            food.append([elem, x**2 + y**2])
                        vision[t, 0] = d_e_type
                        vision[t, 1] = elem.energy
                        vision[t, 2] = elem.signal * int(d_e_type == 1)
                # process NN
                action = entity.step(vision)
                new_pos = (  # used later
                    round(ind[0] + action[0] * self.speed[entity.type]) % self.grid_size[0],
                    round(ind[1] + action[1] * self.speed[entity.type]) % self.grid_size[1]
                )
                # handle newborn
                if entity.energy >= self.energy[entity.type][1]:
                    entity.energy /= 2
                    child = Entity(
                        entity.type,
                        entity.network.child(self.mod_scale),
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
                        possibles_pos = []
                        for dx in range(-self.vision[entity.type], self.vision[entity.type] + 1):
                            for dy in range(-self.vision[entity.type], self.vision[entity.type] + 1):
                                x = (new_pos[0] + dx) % self.grid_size[0]
                                y = (new_pos[1] + dy) % self.grid_size[1]
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
                    possibles_pos = []
                    for dx in range(-self.vision[entity.type], self.vision[entity.type] + 1):
                        for dy in range(-self.vision[entity.type], self.vision[entity.type] + 1):
                            x = (new_pos[0] + dx) % self.grid_size[0]
                            y = (new_pos[1] + dy) % self.grid_size[1]
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
