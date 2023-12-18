import tkinter as tk
from PIL import Image, ImageTk
from simulation import Simulation
import numpy as np

# Rock 0
# Paper 1
# Scissors 2

HIDDEN_NEURONS = []
INITIAL_POPULATION = 20  # per entity type
MAP_SIZE = (30, 30)
TILE_SIZE = 15

class Sim:
    def __init__(self):
        self.running = True
        self.data = {
            'speed': (1, 1, 1),
            'damage': (8, 8, 8),
            'steal': (0.7, 0.7, 0.7),
            'energy': ((70, 106), (70, 106), (70, 106)),  # default energy, required energy to produce a child
            'loss_factor': (0.095, 0.095, 0.095),
            'vision': (12, 12, 12)
        }
        self.simulation = Simulation(MAP_SIZE, INITIAL_POPULATION, HIDDEN_NEURONS, self.data)
        self.log_0 = []
        self.log_1 = []
        self.log_2 = []
    
    def step(self):
        self.simulation.step()
        self.log_0.append(0)
        self.log_1.append(0)
        self.log_2.append(0)
        for i, e in np.ndenumerate(self.simulation.map):
            if e is not None:
                self.log_0[-1] += int(e.type == 0)
                self.log_1[-1] += int(e.type == 1)
                self.log_2[-1] += int(e.type == 2)
        if self.log_0[-1] + self.log_1[-1] + self.log_2[-1] == 0:
            self.running = False

class SimulationUI:
    def __init__(self, window_title, window_icon, map_size, tile_size):
        self.simulation = Simulation(map_size, 50, [10, 10], ) # type: ignore
        self.window = tk.Tk()
        self.window.title(window_title) # titre de la fenêtre
        icon_photo = ImageTk.PhotoImage(Image.open(window_icon))
        self.window.wm_iconphoto(False, icon_photo) # icone de la fenêtre
        self.canvas_frame = tk.Frame()
        self.menu_frame = tk.Frame()
        self.canvas_frame.pack()
        self.menu_frame.pack()
        self.canvas = tk.Canvas(self.canvas_frame, width=map_size[0] * tile_size, height=map_size[1] * tile_size)
        self.btn_run_sim = tk.Button(self.menu_frame, text='Lancer la simulation', command=self.run_simulation)
    
    def show(self):
        self.window.mainloop()
    
    def run_simulation(self):


ui = SimulationUI('PyRPS', 'assets/rock.png', (50, 50), 10)
ui.show()