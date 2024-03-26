from simulation import Simulation, Entity
from side_panel import SimulationControl, SimulationSettings
import customtkinter as ctk
import numpy as np
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import json
import os.path
from time import monotonic


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
        self.settings = SimulationSettings(master=self.sidebar_menu, params_pointer=self.config['sim'])
        self.settings.grid(row=1, column=0, stick='nsew', padx=12, pady=(6, 12))
        self.sidebar_menu.grid(row=0, column=0, stick='nsew')
        self.canvas = CanvasFrame(master=self, tile_size=config['sim']['tile_size'], grid_size=config['sim']['grid_size'])
        self.canvas.grid(row=0, column=1)

        self.sim_running = False
        self.request_sim_stop = False
        self.has_reset = True
        self.tile_size = config['sim']['tile_size']
        self.sim_delta_time = config['sim']['delta_time']
        self.settings.delta_time_var.trace_add('write', self.update_delta_time)
        self.sim = None
        self.sim_log_t = []
        self.sim_log_0 = []
        self.sim_log_1 = []
        self.sim_log_2 = []
        self.menu.on_run(self.launch_sim)
        self.menu.on_stop(self.stop_sim)
        self.menu.on_reset(self.reset_sim)
        self.menu.on_step(self.step_sim)
        self.menu.on_show_plot(self.show_plot)
        self.init_plot()
        self.settings.live_plotting_var.trace_add('write', self.update_live_plotting)

        self.protocol("WM_DELETE_WINDOW", self.on_app_close)
    
    def update_delta_time(self, *_):
        self.sim_delta_time = self.settings.delta_time_var.get()

    def on_app_close(self):
        self.stop_sim()
        self.quit()
    
    def update_live_plotting(self, *_):
        self.live_plotting = self.settings.live_plotting_var.get()

    def init_plot(self, *_):
        self.plot_opened = False
        self.plot = plt.figure()
        self.plot.canvas.mpl_connect('close_event', self.init_plot)

    def legend_plot(self):
        plt.title('Evolution des populations au cours du temps')
        plt.legend(["Total", "Feuille", "Pierre", "Ciseaux", 'Moyenne'])
        plt.grid(True)
        plt.xlabel("Temps")
        plt.ylabel("Nombre d'individus")

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
        if self.has_reset:
            return
        self.has_reset = True
        self.stop_sim()
        self.backup_logs()
        self.clear_canvas()

    def backup_logs(self):
        self.sim_log_t = self.sim.log_t.copy()
        self.sim_log_0 = self.sim.log_0.copy()
        self.sim_log_1 = self.sim.log_1.copy()
        self.sim_log_2 = self.sim.log_2.copy()

    def show_plot(self):
        self.plot_opened = True
        if self.has_reset:
            log_t = self.sim_log_t
            log_0 = self.sim_log_0
            log_1 = self.sim_log_1
            log_2 = self.sim_log_2
        else:
            log_t = self.sim.log_t
            log_0 = self.sim.log_0
            log_1 = self.sim.log_1
            log_2 = self.sim.log_2
        
        log_mean = np.mean([log_0, log_1, log_2], axis=0)

        iters = list(range(len(log_t)))
        self.plot.clear()
        self.legend_plot()
        plt.plot(iters, log_t, color="black")
        plt.plot(iters, log_0, color="red")
        plt.plot(iters, log_1, color="green")
        plt.plot(iters, log_2, color="blue")
        plt.plot(iters, log_mean, color="grey", linestyle="--")
        self.plot.show()

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
        pop_size = min(cfg['pop_size'], cfg['grid_size'][0] * cfg['grid_size'][1])
        self.sim = Simulation(cfg['grid_size'], pop_size, cfg['layers'], cfg['data'])
        self.canvas.change_size(cfg['grid_size'], cfg['tile_size'])
        self.tile_size = cfg['tile_size']
        self.live_plotting = cfg['live_plotting']
        self.load_entity_assets()

    def run_sim(self):
        start_time = monotonic()
        if self.request_sim_stop or not self.sim_running:
            self.request_sim_stop = False
            self.sim_running = False
            return
        self.sim_running = self.sim.step()
        if not self.sim_running: # plus aucune entité en vie
            self.reset_sim()
        if self.plot_opened and self.live_plotting:
            self.show_plot()
        self.update_canvas()
        time_taken = int((monotonic() - start_time) * 1000) # en ms
        next_run_after = min(max(self.sim_delta_time - time_taken, 1), self.sim_delta_time)
        self.after(next_run_after, self.run_sim)

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
