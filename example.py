from simulation import *
import matplotlib.pyplot as plt
import pygame


HIDDEN_NEURONS = [10, 10]
INITIAL_POPULATION = 1000  # per entity type
MAP_SIZE = (256, 256)


class Sim:
	def __init__(self, size):
		pygame.init()
		self.size = size
		self.running = True
		self.screen = pygame.display.set_mode((2*self.size[0], 2*self.size[1]))
		# setup simulation
		data = {
			'speed': (4, 4, 4),
			'damage': (10, 10, 10),
			'steal': (0.6, 0.6, 0.6),
			'range': (2, 2, 2),
			'energy': ((80, 120), (80, 120), (80, 120)),  # default energy, required energy to produce a child
			'loss_factor': (0.01, 0.01, 0.01),
			'vision': (5, 5, 5)
		}
		self.simulation = Simulation(self.size, INITIAL_POPULATION, HIDDEN_NEURONS, data)
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
