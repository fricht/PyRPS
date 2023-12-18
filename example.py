from simulation import *
import matplotlib.pyplot as plt
import tkinter as tk


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
		self.quit_button = tk.Button(self.window, text='Quitter', command=self.exit)
		self.quit_button.pack()
		# setup simulation
		self.data = {
			'speed': (4, 4, 4),
			'damage': (8, 8, 8),
			'steal': (0.7, 0.7, 0.7),
			'energy': ((70, 106), (70, 106), (70, 106)),  # default energy, required energy to produce a child
			'loss_factor': (0.095, 0.095, 0.095),
			'vision': (12, 12, 12)
		}
		self.simulation = Simulation(map_size, INITIAL_POPULATION, HIDDEN_NEURONS, self.data)
		self.log_0 = []
		self.log_1 = []
		self.log_2 = []

	def load_image(self, path, size):
		img = tk.PhotoImage(file=path)
		subsample_x = int(img.width() / size)
		subsample_y = int(img.height() / size)
		return img.subsample(subsample_x, subsample_y)
	
	def init_canvas(self, map_size, tile_size):
		width = map_size[0] * tile_size
		height = map_size[1] * tile_size
		return tk.Canvas(self.window, width=width, height=height)

	def exit(self):
		self.window.quit()
		self.running = False

	def update(self):
		self.simulation.step()
		self.log_0.append(0)
		self.log_1.append(0)
		self.log_2.append(0)
		self.canvas.delete('all')
		for i, e in np.ndenumerate(self.simulation.map):
			if e is not None:
				self.log_0[-1] += int(e.type == 0)
				self.log_1[-1] += int(e.type == 1)
				self.log_2[-1] += int(e.type == 2)
				img = self.assets['rock']
				if e.type == 1:
					img = self.assets['paper']
				elif e.type == 2:
					img = self.assets['scissors']
				self.canvas.create_image(i[0] * self.tile_size, i[1] * self.tile_size, image=img, anchor='nw')
		if self.log_0[-1] + self.log_1[-1] + self.log_2[-1] == 0:
			self.running = False

	def run(self):
		self.window.mainloop()
		print('after mainloop')
		while self.running:
			self.update()
		iters = list(range(self.simulation.tick))
		plt.plot(iters, self.log_0, color="red")
		plt.plot(iters, self.log_1, color="green")
		plt.plot(iters, self.log_2, color="blue")
		plt.legend(["red population", "green population", "blue population"])
		plt.grid(True)
		plt.xlabel("Ticks")
		plt.ylabel("Amount")
		plt.show()


Sim(tuple(MAP_SIZE), TILE_SIZE).run()
