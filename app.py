import customtkinter as ctk
import numpy as np
from simulation import Simulation, EntityTypes
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import json

class MenuFrame(ctk.CTkFrame):
    '''
    Classe pour gérer les boutons de contrôle de la simulation
    '''
    def __init__(self, master):
        super().__init__(master=master, fg_color=master.cget('fg_color'))

        self.btn_run = ctk.CTkButton(master=self, text='Lancer la simulation', bg_color=self.cget('fg_color'))
        self.btn_run.grid(padx=20, pady=(20, 10))
        self.btn_stop = ctk.CTkButton(master=self, text='Stopper la simulation', bg_color=self.cget('fg_color'))
        self.btn_stop.grid(padx=20, pady=10)
        self.btn_step = ctk.CTkButton(master=self, text='Step', bg_color=self.cget('fg_color'))
        self.btn_step.grid(padx=20, pady=10)
        self.btn_reset = ctk.CTkButton(master=self, text='Réinitialiser', bg_color=self.cget('fg_color'))
        self.btn_reset.grid(padx=20, pady=(10, 20))

    def on_run(self, fn):
        self.btn_run.configure(command=fn)

    def on_stop(self, fn):
        self.btn_stop.configure(command=fn)

    def on_reset(self, fn):
        self.btn_reset.configure(command=fn)
    
    def on_step(self, fn):
        self.btn_step.configure(command=fn)

class CanvasFrame(ctk.CTkFrame):
    '''
    Classe pour gérer l'affichage de la simulation
    '''
    def __init__(self, master, grid_size, tile_size):
        super().__init__(master=master, fg_color=master.cget('fg_color'))

        self.canvas = ctk.CTkCanvas(master=self, width=grid_size[0] * tile_size, height=grid_size[1] * tile_size)
        self.canvas.grid(padx=10, pady=10)

class App(ctk.CTk):
    '''
    Classe principale de l'application
    '''
    def __init__(self, config):
        super().__init__()

        self.config = config

        ctk.set_appearance_mode('dark')

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.title('PyRPS Simulation')
        self.wm_iconbitmap()
        self.iconphoto(True, ImageTk.PhotoImage(file='assets/logo.png'))

        self.menu = MenuFrame(master=self)
        self.menu.grid(row=0, column=0, stick='nsew')
        self.canvas = CanvasFrame(master=self, tile_size=config['sim']['tile_size'], grid_size=config['sim']['grid_size'])
        self.canvas.grid(row=0, column=1)

        rock_path = 'assets/the_rock.png' if config['easter_egg'] else 'assets/rock.png'
        self.assets = {
            'rock': self.load_image(rock_path, config['sim']['tile_size']),
            'paper': self.load_image('assets/paper.png', config['sim']['tile_size']),
            'scissors': self.load_image('assets/scissors.png', config['sim']['tile_size'])
        }

        self.sim_running = False
        self.request_sim_stop = False
        self.sim_grid_size = config['sim']['grid_size']
        self.tile_size = config['sim']['tile_size']
        self.sim_pop_size = config['sim']['pop_size']
        self.sim_layers = config['sim']['layers']
        self.sim_data = config['sim']['data']
        self.sim_delta_time = config['sim']['delta_time']
        self.sim = Simulation(tuple(config['sim']['grid_size']), config['sim']['pop_size'], list(config['sim']['layers']), config['sim']['data'])
        self.menu.on_run(self.launch_sim)
        self.menu.on_stop(self.stop_sim)
        self.menu.on_reset(self.reset_sim)
        self.menu.on_step(self.step_sim)

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
                if e.type == EntityTypes.PAPER:
                    img = self.assets['paper']
                if e.type == EntityTypes.ROCK:
                    img = self.assets['rock']
                elif e.type == EntityTypes.SCISSORS:
                    img = self.assets['scissors']
                self.canvas.canvas.create_image(i[0] * self.tile_size, i[1] * self.tile_size, image=img, anchor='nw')

    def reset_sim(self):
        self.stop_sim()
        self.clear_canvas()
        self.show_plot()
        self.sim.reset()
    
    def show_plot(self):
        iters = list(range(len(self.sim.log_t)))
        plt.clf()
        plt.title('Evolution des populations au cours du temps')
        plt.plot(iters, self.sim.log_t, color="grey")
        plt.plot(iters, self.sim.log_0, color="red")
        plt.plot(iters, self.sim.log_1, color="green")
        plt.plot(iters, self.sim.log_2, color="blue")
        plt.legend(["Total", "Feuille", "Pierre", "Ciseaux"])
        plt.grid(True)
        plt.xlabel("Temps")
        plt.ylabel("Nombre d'individus")
        plt.show()
    
    def clear_canvas(self):
        self.canvas.canvas.delete('all')
    
    def launch_sim(self):
        if self.sim_running:
            return
        self.run_sim()

    def run_sim(self):
        if self.request_sim_stop:
            self.request_sim_stop = False
            self.sim_running = False
            return
        self.sim_running = self.sim.step()
        self.update_canvas()
        self.after(self.sim_delta_time, self.run_sim)

    def stop_sim(self):
        if self.sim_running:
            self.request_sim_stop = True
    
    def step_sim(self):
        if self.sim_running:
            return
        self.sim.step()
        self.update_canvas()

with open('config.json', 'r') as f:
    config = json.load(f)

app = App(config)
app.mainloop()
