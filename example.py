from simulation import *
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
		data = {
			'speed': (6, 4, 3),
			'damage': (8, 10, 11),
			'steal': (0.7, 0.6, 0.74),
			'energy': ((60, 100), (80, 120), (100, 150)),  # default energy, required energy to produce a child
			'loss_factor': (0.08, 0.08, 0.08),
			'vision': (7, 7, 8)
		}
		data_eq = {
			'speed': (4, 4, 4),
			'damage': (8, 8, 8),
			'steal': (0.7, 0.7, 0.7),
			'energy': ((70, 102), (70, 102), (70, 102)),  # default energy, required energy to produce a child
			'loss_factor': (0.082, 0.082, 0.082),
			'vision': (12, 12, 12)
		}
		self.simulation = Simulation(self.size, INITIAL_POPULATION, HIDDEN_NEURONS, data_eq)
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
					255 * int(e.type == 0),
					255 * int(e.type == 1),
					255 * int(e.type == 2)
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
