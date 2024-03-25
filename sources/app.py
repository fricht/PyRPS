from simulation import Simulation, Entity
from side_panel import SimulationControl, SimulationSettings
import customtkinter as ctk
import numpy as np
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import json
import os.path


PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')


class CanvasFrame(ctk.CTkFrame):
    '''
    Classe pour gérer l'affichage de la simulation
    '''
    def __init__(self, master, grid_size, tile_size):
        super().__init__(master=master, fg_color=master.cget('fg_color'))

        self.canvas = ctk.CTkCanvas(master=self, width=grid_size[0] * tile_size, height=grid_size[1] * tile_size)
        self.canvas.pack(padx=10, pady=10)
    
    def clear(self):
        self.canvas.delete('all')
    
    def change_size(self, grid_size, tile_size):
        self.canvas.config(width=grid_size[0] * tile_size, height=grid_size[1] * tile_size)
    
    def draw_entity(self, x, y, img):
        self.canvas.create_image(x + 2, y + 2, image=img, anchor='nw')


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
        self.assets_path = os.path.join(PROJECT_DIR, 'assets')
        logo_path = os.path.join(self.assets_path, 'logo.png')
        self.wm_iconbitmap()
        self.iconphoto(True, ImageTk.PhotoImage(file=logo_path))

        self.sidebar_menu = ctk.CTkFrame(self)
        self.menu = SimulationControl(master=self.sidebar_menu)
        self.menu.grid(row=0, column=0, stick='nsew', padx=12, pady=(12, 6))
        self.settings = SimulationSettings(master=self.sidebar_menu, params_pointer=self.config)
        self.settings.grid(row=1, column=0, stick='nsew', padx=12, pady=(6, 12))
        self.sidebar_menu.grid(row=0, column=0, stick='nsew')
        self.canvas = CanvasFrame(master=self, tile_size=config['sim']['tile_size'], grid_size=config['sim']['grid_size'])
        self.canvas.grid(row=0, column=1)

        self.sim_running = False
        self.request_sim_stop = False
        self.has_reset = True
        self.sim_grid_size = config['sim']['grid_size']
        self.tile_size = config['sim']['tile_size']
        self.sim_pop_size = config['sim']['pop_size']
        self.sim_layers = config['sim']['layers']
        self.sim_data = config['sim']['data']
        self.sim_delta_time = config['sim']['delta_time']
        self.sim = Simulation(config['sim']['grid_size'], config['sim']['pop_size'], config['sim']['layers'], config['sim']['data'])
        self.menu.on_run(self.launch_sim)
        self.menu.on_stop(self.stop_sim)
        self.menu.on_reset(self.reset_sim)
        self.menu.on_step(self.step_sim)
        self.menu.on_show_plot(self.show_plot)

    def load_entity_assets(self):
        rock_path = os.path.join(self.assets_path, 'the_rock.png' if self.config['easter_egg'] else 'rock.png')
        self.assets = {
            'rock': self.load_image(rock_path, self.tile_size),
            'paper': self.load_image(os.path.join(self.assets_path, 'paper.png'), self.tile_size),
            'scissors': self.load_image(os.path.join(self.assets_path, 'scissors.png'), self.tile_size)
        }

    def load_image(self, path, size):
        img = Image.open(path).resize((size, size))
        return ImageTk.PhotoImage(img)

    def update_canvas(self):
        self.clear_canvas()
        for i, e in np.ndenumerate(self.sim.map):
            if e is not None:
                img = None
                if e.type == Entity.Types.PAPER:
                    img = self.assets['paper']
                if e.type == Entity.Types.ROCK:
                    img = self.assets['rock']
                elif e.type == Entity.Types.SCISSORS:
                    img = self.assets['scissors']
                self.canvas.draw_entity(i[0] * self.tile_size, i[1] * self.tile_size, img)

    def reset_sim(self):
        self.stop_sim()
        self.backup_logs()
        self.has_reset = True
        self.clear_canvas()

    def backup_logs(self):
        self.sim_log_t = self.sim.log_t.copy()
        self.sim_log_0 = self.sim.log_0.copy()
        self.sim_log_1 = self.sim.log_1.copy()
        self.sim_log_2 = self.sim.log_2.copy()

    def show_plot(self):
        log_t = self.sim.log_t
        log_0 = self.sim.log_0
        log_1 = self.sim.log_1
        log_2 = self.sim.log_2
        if self.has_reset:
            log_t = self.sim_log_t
            log_0 = self.sim_log_0
            log_1 = self.sim_log_1
            log_2 = self.sim_log_2

        iters = list(range(len(log_t)))
        plt.clf()
        plt.title('Evolution des populations au cours du temps')
        plt.plot(iters, log_t, color="grey")
        plt.plot(iters, log_0, color="red")
        plt.plot(iters, log_1, color="green")
        plt.plot(iters, log_2, color="blue")
        plt.legend(["Total", "Feuille", "Pierre", "Ciseaux"])
        plt.grid(True)
        plt.xlabel("Temps")
        plt.ylabel("Nombre d'individus")
        plt.show()

    def clear_canvas(self):
        self.canvas.clear()

    def launch_sim(self):
        if self.sim_running:
            return
        if self.has_reset:
            self.apply_sim_settings()
        self.sim_running = True
        self.run_sim()
    
    def apply_sim_settings(self):
        self.has_reset = False
        cfg = self.settings.get_data()
        self.sim = Simulation(cfg['sim']['grid_size'], cfg['sim']['pop_size'], cfg['sim']['layers'], cfg['sim']['data'])
        self.canvas.change_size(cfg['sim']['grid_size'], cfg['sim']['tile_size'])
        self.tile_size = cfg['sim']['tile_size']
        self.load_entity_assets()

    def run_sim(self):
        if self.request_sim_stop or not self.sim_running:
            self.request_sim_stop = False
            self.sim_running = False
            return
        self.sim_running = self.sim.step()
        if not self.sim_running:
            self.reset_sim()
        self.update_canvas()
        self.after(self.sim_delta_time, self.run_sim)

    def stop_sim(self):
        if self.sim_running:
            self.request_sim_stop = True

    def step_sim(self):
        if self.sim_running:
            return
        if self.has_reset:
            self.apply_sim_settings()
        self.sim.step()
        self.update_canvas()

config_path = os.path.join(PROJECT_DIR, 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

app = App(config)
app.mainloop()
