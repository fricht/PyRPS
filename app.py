import customtkinter as ctk
import numpy as np
from simulation import Simulation
import tkinter as tk
from PIL import ImageTk, Image

class MenuFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master=master, fg_color=master.cget('fg_color'))

        self.run, self.stop = None, None

        self.btn_run = ctk.CTkButton(master=self, text='Lancer la simulation', bg_color=self.cget('fg_color'))
        self.btn_run.grid(padx= 20, pady=(20, 10))
        self.btn_stop = ctk.CTkButton(master=self, text='Stopper la simulation', bg_color=self.cget('fg_color'))
        self.btn_stop.grid(padx=20, pady=10)
        self.btn_reset = ctk.CTkButton(master=self, text='Réinitialiser', bg_color=self.cget('fg_color'))
        self.btn_reset.grid(padx=20, pady=10)
        self.btn_step = ctk.CTkButton(master=self, text='Step', bg_color=self.cget('fg_color'))
        self.btn_step.grid(padx=20, pady=(10, 20))

    def on_run(self, fn):
        self.btn_run.configure(command=fn)

    def on_stop(self, fn):
        self.btn_stop.configure(command=fn)

    def on_reset(self, fn):
        self.btn_reset.configure(command=fn)

class CanvasFrame(ctk.CTkFrame):
    def __init__(self, master, grid_size, tile_size):
        super().__init__(master=master, fg_color=master.cget('fg_color'))

        self.canvas = ctk.CTkCanvas(master=self, width=grid_size[0] * tile_size, height=grid_size[1] * tile_size)
        self.canvas.grid(padx=10, pady=10)

class App(ctk.CTk):
    def __init__(self, window_title, grid_size, tile_size, pop_size, layers, sim_data):
        super().__init__()

        ctk.set_appearance_mode('dark')

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.title(window_title)
        self.wm_iconbitmap()
        self.iconphoto(True, ImageTk.PhotoImage(file='assets/the_rock.png'))

        self.menu = MenuFrame(master=self)
        self.menu.grid(row=0, column=0, stick='nsew')
        self.canvas = CanvasFrame(master=self, tile_size=tile_size, grid_size=grid_size)
        self.canvas.grid(row=0, column=1)

        self.assets = {
            'rock': self.load_image('assets/the_rock.png', tile_size),
            'paper': self.load_image('assets/paper.png', tile_size),
            'scissors': self.load_image('assets/scissors.png', tile_size)
        }

        self.sim_running = True
        self.sim_grid_size = grid_size
        self.tile_size = tile_size
        self.sim_pop_size = pop_size
        self.sim_layers = layers
        self.sim_data = sim_data
        self.sim = Simulation(tuple(grid_size), pop_size, list(layers), sim_data)
        self.menu.on_run(self.run_sim)
        self.menu.on_stop(self.stop_sim)
        self.menu.on_reset(self.reset_sim)
    
    def load_image(self, path, size):
        img = Image.open(path).resize((size, size))
        return ImageTk.PhotoImage(img)
    
    def on_reset(self, fn):
        self.menu.on_reset(fn)

    def update_canvas(self):
        self.clear_canvas()
        for i, e in np.ndenumerate(self.sim.map):
            if e is not None:
                img = None
                if e.type == 0:
                    img = self.assets['paper']
                if e.type == 1:
                    img = self.assets['rock']
                elif e.type == 2:
                    img = self.assets['scissors']
                self.canvas.canvas.create_image(i[0] * self.tile_size, i[1] * self.tile_size, image=img, anchor='nw')
    
    def reset_sim(self):
        self.stop_sim()
        self.clear_canvas()
        self.sim = Simulation(tuple(self.sim_grid_size), self.sim_pop_size, list(self.sim_layers), self.sim_data) # TODO améliorer le reset
    
    def clear_canvas(self):
        self.canvas.canvas.delete('all')
    
    def run_sim(self):
        if self.sim_running:
            self.sim_running = self.sim.step()
            self.update_canvas()
            self.after(250, self.run_sim)
        else:
            self.sim_running = True

    def stop_sim(self):
        self.sim_running = False

    def setup_sim(self):
        pass


data = {
    'speed': (1, 1, 1),
    'damage': (8, 8, 8),
    'steal': (0.7, 0.7, 0.7),
    'energy': ((70, 101), (70, 101), (70, 101)),  # default energy, required energy to produce a child
    'loss_factor': (0.06, 0.06, 0.06),
    'vision': (12, 12, 12),
    'range': (5, 5, 5)
}

app = App(window_title='PyRPS Simulation', grid_size=(30, 30), tile_size=20, pop_size=30, layers=(10, 10), sim_data=data)
app.mainloop()