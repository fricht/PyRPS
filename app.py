import customtkinter as ctk
import numpy as np
from simulation import Simulation, Entity
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import json


class HelpWondow(ctk.CTkToplevel):
    def __init__(self, text, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Aide")
        ctk.CTkLabel(self, text=text).pack(padx=20, pady=20)


class FrameTitle(ctk.CTkLabel):
    def __init__(self, master, text):
        super().__init__(master=master, text=text, font=('Arial', 16))


class MenuFrame(ctk.CTkFrame):
    '''
    Classe pour gérer les boutons de contrôle de la simulation
    '''
    def __init__(self, master):
        super().__init__(master=master, fg_color=master.cget('fg_color'))

        FrameTitle(master=self, text='Simulation').grid(column=0, row=0, columnspan=2)

        self.btn_run = ctk.CTkButton(master=self, text='Lancer la simulation', bg_color=self.cget('fg_color'))
        self.btn_run.grid(column=0, row=1, padx=5, pady=5)
        self.btn_stop = ctk.CTkButton(master=self, text='Stopper la simulation', bg_color=self.cget('fg_color'))
        self.btn_stop.grid(column=1, row=1, padx=5, pady=5)
        self.btn_step = ctk.CTkButton(master=self, text='Step', bg_color=self.cget('fg_color'))
        self.btn_step.grid(column=0, row=2, padx=5, pady=5)
        self.btn_reset = ctk.CTkButton(master=self, text='Réinitialiser', bg_color=self.cget('fg_color'))
        self.btn_reset.grid(column=1, row=2, padx=5, pady=5)

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


class SingleAttributeEdit(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master=master, fg_color=master.cget('fg_color'))

        ctk.CTkLabel(self, text="Vitesse").grid(row=0, column=0)
        self.speed_var = ctk.IntVar()
        ctk.CTkSlider(self, from_=1, to=10, variable=self.speed_var, number_of_steps=9).grid(row=0, column=1)
        ctk.CTkLabel(self, textvariable=self.speed_var).grid(row=0, column=2)

        ctk.CTkLabel(self, text="Dégats").grid(row=1, column=0)
        self.damage_var = ctk.IntVar()
        ctk.CTkSlider(self, from_=1, to=20, variable=self.damage_var, number_of_steps=19).grid(row=1, column=1)
        ctk.CTkLabel(self, textvariable=self.damage_var).grid(row=1, column=2)

        ctk.CTkLabel(self, text="Vol d'énergie").grid(row=2, column=0)
        self.steal_var = ctk.DoubleVar()
        ctk.CTkSlider(self, from_=0, to=1, variable=self.steal_var).grid(row=2, column=1)
        ctk.CTkLabel(self, textvariable=self.steal_var).grid(row=2, column=2)

        ctk.CTkLabel(self, text="Énergie de naissance").grid(row=3, column=0)
        self.energy_def_var = ctk.IntVar()
        ctk.CTkSlider(self, from_=1, to=200, variable=self.energy_def_var, number_of_steps=199).grid(row=3, column=1)
        ctk.CTkLabel(self, textvariable=self.energy_def_var).grid(row=3, column=2)

        ctk.CTkLabel(self, text="Énergie pour reproduction").grid(row=4, column=0)
        self.energy_child_var = ctk.IntVar()
        ctk.CTkSlider(self, from_=1, to=200, variable=self.energy_child_var, number_of_steps=199).grid(row=4, column=1)
        ctk.CTkLabel(self, textvariable=self.energy_child_var).grid(row=4, column=2)

        ctk.CTkLabel(self, text="Facteur de vieillissement").grid(row=5, column=0) # loss_factor
        self.aging_var = ctk.DoubleVar()
        ctk.CTkSlider(self, from_=0, to=1, variable=self.aging_var).grid(row=5, column=1)
        ctk.CTkLabel(self, textvariable=self.aging_var).grid(row=5, column=2)

        ctk.CTkLabel(self, text="Vision").grid(row=6, column=0)
        self.vision_var = ctk.IntVar()
        ctk.CTkSlider(self, from_=1, to=20, variable=self.vision_var, number_of_steps=19).grid(row=6, column=1)
        ctk.CTkLabel(self, textvariable=self.vision_var).grid(row=6, column=2)

        ctk.CTkLabel(self, text="Portée").grid(row=7, column=0)
        self.range_var = ctk.IntVar()
        ctk.CTkSlider(self, from_=1, to=20, variable=self.range_var, number_of_steps=19).grid(row=7, column=1)
        ctk.CTkLabel(self, textvariable=self.range_var).grid(row=7, column=2)

    def get_values(self):
        return {
            "speed": self.speed_var.get(),
            "damage": self.damage_var.get(),
            "steal": self.steal_var.get(),
            "energy": [self.energy_def_var.get(), self.energy_child_var.get()],
            "loss_factor": self.aging_var.get(),
            "vision": self.vision_var.get(),
            "range": self.range_var.get()
        }

    def set_values(self, data):
        self.speed_var.set(data['speed'])
        self.damage_var.set(data['damage'])
        self.steal_var.set(data['steal'])
        self.energy_def_var.set(data['energy'][0])
        self.energy_child_var.set(data['energy'][1])
        self.aging_var.set(data['loss_factor'])
        self.vision_var.set(data['vision'])
        self.range_var.set(data['range'])


class EntityAttributes(ctk.CTkFrame):
    def __init__(self, master, params_pointer):
        super().__init__(master=master, fg_color=master.cget('fg_color'))
        self.params = params_pointer

        self.help_window = None

        FrameTitle(master=self, text="Paramètres").pack()

        ctk.CTkButton(self, text='aide', command=self.on_help).pack(pady=10)

        self.actions_frame = ctk.CTkFrame(self)
        ctk.CTkButton(self.actions_frame, text="Reset", command=self.reset_params).grid(row=0, column=0, padx=10, pady=10)
        ctk.CTkButton(self.actions_frame, text="Sauver", command=self.save_params).grid(row=0, column=1, padx=10, pady=10)
        self.actions_frame.pack()

        self.params_select = ctk.CTkTabview(self, bg_color=self.cget('fg_color'))
        self.params_select.pack()

        ctk.CTkLabel(self.params_select.add("Général"), text="Dommage, ça marche pas encore...").pack()

        self.rock_settings = SingleAttributeEdit(self.params_select.add("Pierre"))
        self.paper_settings = SingleAttributeEdit(self.params_select.add("Feuille"))
        self.sissors_settings = SingleAttributeEdit(self.params_select.add("Ciseaux"))
        self.rock_settings.pack()
        self.paper_settings.pack()
        self.sissors_settings.pack()

    def on_help(self):
        if self.help_window is None or not self.help_window.winfo_exists():
            self.toplevel_window = HelpWondow("Ceci est un message...\net il saute des lignes !!!")  # create window if its None or destroyed
        else:
            self.toplevel_window.focus()  # if window exists focus it

    def reset_params(self):
        rock = {}
        paper = {}
        sissors = {}
        for k, v in self.params['sim']['data'].items():
            if type(v) in {tuple, list}:
                paper[k] = v[0]
                rock[k] = v[1]
                sissors[k] = v[2]
        self.rock_settings.set_values(rock)
        self.paper_settings.set_values(paper)
        self.sissors_settings.set_values(sissors)

    def get_data(self):
        rock = self.rock_settings.get_values()
        paper = self.paper_settings.get_values()
        sissors = self.sissors_settings.get_values()
        data = {k: [paper[k], rock[k], sissors[k]] for k in paper.keys() & rock.keys() & sissors.keys()}
        data['mod_scale'] = 0.002 # TODO : implement this in general settings
        return {
            "easter_egg": True,
            "sim": {
                "delta_time": 1,
                "grid_size": [30, 30],
                "tile_size": 10,
                "pop_size": 20,
                "layers": [10, 10],
                "data": data
            }
        }

    def save_params(self):
        self.params = self.get_data()


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

        self.sidebar_menu = ctk.CTkFrame(self)
        self.menu = MenuFrame(master=self.sidebar_menu)
        self.menu.pack()
        self.settings = EntityAttributes(master=self.sidebar_menu, params_pointer=self.config)
        self.settings.pack()
        self.settings.reset_params()
        self.sidebar_menu.grid(row=0, column=0, stick='nsew')
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
        self.sim = Simulation(config['sim']['grid_size'], config['sim']['pop_size'], config['sim']['layers'], config['sim']['data'])
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
                if e.type == Entity.Types.PAPER:
                    img = self.assets['paper']
                if e.type == Entity.Types.ROCK:
                    img = self.assets['rock']
                elif e.type == Entity.Types.SCISSORS:
                    img = self.assets['scissors']
                self.canvas.canvas.create_image(i[0] * self.tile_size, i[1] * self.tile_size, image=img, anchor='nw')

    def reset_sim(self):
        self.stop_sim()
        self.clear_canvas()
        self.show_plot()
        # using the user config
        cfg = self.settings.get_data()
        self.sim = Simulation(cfg['sim']['grid_size'], cfg['sim']['pop_size'], cfg['sim']['layers'], cfg['sim']['data'])

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
