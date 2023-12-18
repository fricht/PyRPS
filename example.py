import importlib.util
try:
	# replace this with your compiled file
	spec = importlib.util.spec_from_file_location(
	"simulation", "compiled.file.here"
	)
	simulation = importlib.util.module_from_spec(spec)
except ImportError:
	print('WARNING : error while importing compiled file')
	simulation = importlib.import_module("simulation.py")
import numpy as np
import simulation
import matplotlib.pyplot as plt
import pygame


HIDDEN_NEURONS = []
INITIAL_POPULATION = 500  # per entity type
MAP_SIZE = (256, 256)


class Sim:
	def __init__(self, size):
		# pygame stuff
		pygame.init()
		self.size = size
		self.running = True
		self.screen = pygame.display.set_mode((2*self.size[0], 2*self.size[1]))
		# setup simulation
		self.data = {
			'speed': (4, 4, 4),
			'damage': (8, 8, 8),
			'steal': (0.64, 0.64, 0.64),
			'energy': ((70, 106), (70, 106), (70, 106)),  # default energy, required energy to produce a child
			'loss_factor': (0.095, 0.095, 0.095),
			'vision': (12, 12, 12)
		}
		self.simulation = simulation.Simulation(self.size, INITIAL_POPULATION, HIDDEN_NEURONS, self.data)
		self.log_0 = []
		self.log_1 = []
		self.log_2 = []

	def update(self):
		self.screen.fill((10, 10, 10))
		self.simulation.step()
		self.log_0.append(0)
		self.log_1.append(0)
		self.log_2.append(0)
		for i, e in np.ndenumerate(self.simulation.map):
			if e is not None:
				self.log_0[-1] += int(e.type == 0)
				self.log_1[-1] += int(e.type == 1)
				self.log_2[-1] += int(e.type == 2)
				c = (
					min(max(255 * (e.energy / self.data['energy'][0][1]), 50), 255) * int(e.type == 0),
					min(max(255 * (e.energy / self.data['energy'][1][1]), 50), 255) * int(e.type == 1),
					min(max(255 * (e.energy / self.data['energy'][2][1]), 50), 255) * int(e.type == 2)
				)
				self.screen.set_at((2 * i[0], 2 * i[1]), c)
				self.screen.set_at((2 * i[0]+1, 2 * i[1]), c)
				self.screen.set_at((2 * i[0], 2 * i[1]+1), c)
				self.screen.set_at((2 * i[0]+1, 2 * i[1]+1), c)
		if self.log_0[-1] + self.log_1[-1] + self.log_2[-1] == 0:
			self.running = False

	def run(self):
		while self.running:
			self.update()
			pygame.display.flip()
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					self.running = False
		pygame.quit()
		iters = list(range(self.simulation.tick))
		plt.plot(iters, self.log_0, color="red")
		plt.plot(iters, self.log_1, color="green")
		plt.plot(iters, self.log_2, color="blue")
		plt.legend(["red population", "green population", "blue population"])
		plt.grid(True)
		plt.xlabel("Ticks")
		plt.ylabel("Amount")
		plt.show()


Sim(tuple(MAP_SIZE)).run()
