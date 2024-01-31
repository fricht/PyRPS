import numpy as np
import simulation
import matplotlib.pyplot as plt
import pygame


HIDDEN_NEURONS = []
INITIAL_POPULATION = 600  # per entity type
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
			'damage': (10, 10, 10),
			'steal': (0.72, 0.72, 0.72),
			'energy': ((70, 101), (70, 101), (70, 101)),  # default energy, required energy to produce a child
			'loss_factor': (0.08, 0.08, 0.08),
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
