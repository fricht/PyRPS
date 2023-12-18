import importlib.util
try:
	# replace this with your compiled file
	spec = importlib.util.spec_from_file_location(
	"simulation", "./simulation.cpython-310-x86_64-linux-gnu.so"
	)
	simulation = importlib.util.module_from_spec(spec)
except ImportError:
	print('WARNING : error while importing compiled file')
	simulation = importlib.import_module("simulation")
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
			'speed': (5, 5, 5),
			'damage': (8, 8, 8),
			'steal': (0.7, 0.7, 0.7),
			'energy': ((70, 100), (70, 100), (70, 100)),  # default energy, required energy to produce a child
			'loss_factor': (0.07, 0.07, 0.07),
			'vision': (12, 12, 12),
			'range': (5, 5, 5)
		}
		self.simulation = simulation.Simulation(self.size, INITIAL_POPULATION, HIDDEN_NEURONS, self.data)

	def update(self):
		self.screen.fill((0, 0, 0))
		continuing = self.simulation.step()
		for i, e in np.ndenumerate(self.simulation.map):
			if e is not None:
				c = (
					min(max(255 * (e.energy / self.data['energy'][0][1]), 50), 255) * int(e.type == 0),
					min(max(255 * (e.energy / self.data['energy'][1][1]), 50), 255) * int(e.type == 1),
					min(max(255 * (e.energy / self.data['energy'][2][1]), 50), 255) * int(e.type == 2)
				)
				self.screen.set_at((2 * i[0], 2 * i[1]), c)
				self.screen.set_at((2 * i[0]+1, 2 * i[1]), c)
				self.screen.set_at((2 * i[0], 2 * i[1]+1), c)
				self.screen.set_at((2 * i[0]+1, 2 * i[1]+1), c)
		self.running = continuing

	def run(self):
		while self.running:
			self.update()
			pygame.display.flip()
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					self.running = False
		pygame.quit()
		iters = list(range(self.simulation.tick))
		plt.plot(iters, self.simulation.log_t, color="grey")
		plt.plot(iters, self.simulation.log_0, color="red")
		plt.plot(iters, self.simulation.log_1, color="green")
		plt.plot(iters, self.simulation.log_2, color="blue")
		plt.legend(["total", "red population", "green population", "blue population"])
		plt.grid(True)
		plt.xlabel("Ticks")
		plt.ylabel("Amount")
		plt.show()


Sim(tuple(MAP_SIZE)).run()
