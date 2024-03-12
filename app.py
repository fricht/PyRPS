import customtkinter as ctk
import numpy as np
from simulation import Simulation
from PIL import ImageTk, Image
import matplotlib.pyplot as plt

# Paper 0
# Rock 1
# Scissors 2

class MenuFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master=master, fg_color=master.cget('fg_color'))

        self.run_strvar = ctk.StringVar(master=self, value='Lancer la simulation')

        self.btn_run = ctk.CTkButton(master=self, textvariable=self.run_strvar, bg_color=self.cget('fg_color'))
        self.btn_run.grid(padx=20, pady=(20, 10))
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
    def __init__(self, window_title, grid_size, tile_size, pop_size, layers, sim_data, delta_time):
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

        self._sim_running = True
        self.sim_grid_size = grid_size
        self.tile_size = tile_size
        self.sim_pop_size = pop_size
        self.sim_layers = layers
        self.sim_data = sim_data
        self.sim_delta_time = delta_time
        self.sim = Simulation(tuple(grid_size), pop_size, list(layers), sim_data)
        self.has_reset = False
        self.menu.on_run(self.launch_sim)
        self.menu.on_stop(self.stop_sim)
        self.menu.on_reset(self.reset_sim)
    
    @property
    def sim_running(self):
        return self._sim_running
    
    @sim_running.setter
    def sim_running(self, value):
        if value:
            self.menu.run_strvar.set('Relancer la simulation')
        else:
            self.menu.run_strvar.set('Lancer la simulation')
        self._sim_running = value

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
    
    def load_new_sim(self):
        self.sim = Simulation(tuple(self.sim_grid_size), self.sim_pop_size, list(self.sim_layers), self.sim_data)  # TODO améliorer le reset
    
    def reset_sim(self):
        self.has_reset = True
        self.stop_sim()
        self.clear_canvas()
        self.load_new_sim()
        iters = list(range(len(self.sim.log_t)))
        plt.plot(iters, self.sim.log_t, color="grey")
        plt.plot(iters, self.sim.log_0, color="red")
        plt.plot(iters, self.sim.log_1, color="green")
        plt.plot(iters, self.sim.log_2, color="blue")
        plt.legend(["total", "red population", "green population", "blue population"])
        plt.grid(True)
        plt.xlabel("Ticks")
        plt.ylabel("Amount")
        plt.show()
    
    def clear_canvas(self):
        self.canvas.canvas.delete('all')
    
    def launch_sim(self):
        if self.has_reset:
            self.has_reset = False
            self.load_new_sim()
        self.run_sim()

    def run_sim(self):
        if self.sim_running:
            self.sim_running = self.sim.step()
            self.update_canvas()
            self.after(self.sim_delta_time, self.run_sim)
        else:
            self.sim_running = True

    def stop_sim(self):
        self.sim_running = False

data = {
    'speed': (1, 1, 1),
    'damage': (8, 8, 8),
    'steal': (0.7, 0.7, 0.7),
    'energy': ((70, 101), (70, 101), (70, 101)),  # default energy, required energy to produce a child
    'loss_factor': (0.06, 0.06, 0.06),
    'vision': (12, 12, 12),
    'range': (5, 5, 5)
}

app = App(window_title='PyRPS Simulation', grid_size=(30, 30), tile_size=10, pop_size=20, layers=(10, 10), sim_data=data, delta_time=10)
app.mainloop()
