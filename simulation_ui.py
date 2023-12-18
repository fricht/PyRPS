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

class SimWrap:
    def __init__(self):
        self.running = True
        self.data = {
            'speed': (1, 1, 1),
            'damage': (8, 8, 8),
            'steal': (0.7, 0.7, 0.7),
            'energy': ((70, 100), (70, 100), (70, 100)),  # default energy, required energy to produce a child
            'loss_factor': (0.07, 0.07, 0.07),
            'vision': (12, 12, 12),
            'range': (5, 5, 5)
        }
        self.simulation = Simulation(MAP_SIZE, INITIAL_POPULATION, HIDDEN_NEURONS, self.data)
    
    def step(self):
        self.simulation.step()
        if self.simulation.log_0[-1] + self.simulation.log_1[-1] + self.simulation.log_2[-1] == 0:
            self.running = False

class SimulationUI:
    def __init__(self, window_title, window_icon, map_size, canvas_tile_size):
        self.canvas_tile_size = canvas_tile_size
        self.sim_wrap = SimWrap()
        self.window = tk.Tk()
        self.window.title(window_title) # titre de la fenêtre
        icon_photo = ImageTk.PhotoImage(Image.open(window_icon))
        self.window.wm_iconphoto(False, icon_photo) # icone de la fenêtre
        self.canvas_frame = tk.Frame()
        self.menu_frame = tk.Frame()
        self.canvas = tk.Canvas(self.canvas_frame, width=map_size[0] * canvas_tile_size, height=map_size[1] * canvas_tile_size)
        self.btn_run_sim = tk.Button(self.menu_frame, text='Lancer la simulation', command=self.run)
        self.btn_stop_sim = tk.Button(self.menu_frame, text='Stopper la simulation', command=self.stop)
        self.canvas_frame.pack()
        self.menu_frame.pack()
        self.canvas.pack()
        self.btn_run_sim.pack()
        self.btn_stop_sim.pack()
        self.assets = {
            'rock': self.load_image('assets/rock.png', canvas_tile_size),
            'paper': self.load_image('assets/paper.png', canvas_tile_size),
            'scissors': self.load_image('assets/scissors.png', canvas_tile_size)
        }
    
    def load_image(self, path, size):
        img = tk.PhotoImage(file=path)
        subsample_x = int(img.width() / size)
        subsample_y = int(img.height() / size)
        return img.subsample(subsample_x, subsample_y)

    def show(self):
        self.window.mainloop()
    
    def run(self):
        if self.sim_wrap.running:
            self.sim_wrap.step()
            self.update_canvas()
            self.canvas.after(250, self.run)
        else:
            self.sim_wrap.running = True
    
    def stop(self):
        self.sim_wrap.running = False
    
    def update_canvas(self):
        self.canvas.delete('all')
        for i, e in np.ndenumerate(self.sim_wrap.simulation.map):
            if e is not None:
                img = None
                if e.type == 0:
                    img = self.assets['rock']
                if e.type == 1:
                    img = self.assets['paper']
                elif e.type == 2:
                    img = self.assets['scissors']
                self.canvas.create_image(i[0] * self.canvas_tile_size, i[1] * self.canvas_tile_size, image=img, anchor='nw')

ui = SimulationUI('PyRPS', 'assets/rock.png', (50, 50), 10)
ui.show()