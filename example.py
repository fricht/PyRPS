from simulation import Simulation
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np


HIDDEN_NEURONS = []
INITIAL_POPULATION = 20  # per entity type
MAP_SIZE = (30, 30)
TILE_SIZE = 15


class Sim:
	def __init__(self, map_size, tile_size):
		# tkinter stuff
		self.running = True
		self.window = tk.Tk(className='PyRPS')
		self.assets = {
			'rock': self.load_image('assets/rock.png', tile_size),
			'paper': self.load_image('assets/paper.png', tile_size),
			'scissors': self.load_image('assets/scissors.png', tile_size)
		}
		self.tile_size = tile_size
		self.canvas = self.init_canvas(map_size, tile_size)
		self.canvas.pack()
		self.btn_run_sim = tk.Button(text='Run', command=self.run_sim)
		self.btn_stop_sim = tk.Button(text='Stop', command=self.stop_sim)
		self.btn_stop_sim.pack()
		self.btn_run_sim.pack()
		# setup simulation
		self.data = {
			'speed': (1, 1, 1),
			'damage': (8, 8, 8),
			'steal': (0.7, 0.7, 0.7),
			'energy': ((70, 100), (70, 100), (70, 100)),  # default energy, required energy to produce a child
			'loss_factor': (0.07, 0.07, 0.07),
			'vision': (12, 12, 12),
			'range': (5, 5, 5)
		}
		self.simulation = Simulation(map_size, INITIAL_POPULATION, HIDDEN_NEURONS, self.data)

	def load_image(self, path, size):
			img = tk.PhotoImage(file=path)
			subsample_x = int(img.width() / size)
			subsample_y = int(img.height() / size)
			return img.subsample(subsample_x, subsample_y)
	
	def init_canvas(self, map_size, tile_size):
		width = map_size[0] * tile_size
		height = map_size[1] * tile_size
		return tk.Canvas(self.window, width=width, height=height)
	
	def stop_sim(self):
		self.running = False

	def run_sim(self):
		if not self.running:
			self.running = True
		if self.running:
			self.update()
			self.canvas.after(500, self.run_sim)

	def update(self):
		self.simulation.step()
		self.canvas.delete('all')
		for i, e in np.ndenumerate(self.simulation.map):
			if e is not None:
				img = self.assets['rock']
				if e.type == 1:
					img = self.assets['paper']
				elif e.type == 2:
					img = self.assets['scissors']
				self.canvas.create_image(i[0] * self.tile_size, i[1] * self.tile_size, image=img, anchor='nw')

	def run(self):
		self.window.mainloop()
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


Sim(tuple(MAP_SIZE), TILE_SIZE).run()
