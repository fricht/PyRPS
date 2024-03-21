import customtkinter as ctk


class HelpWindowManager:
    '''
    Wrapper pour afficher une fenêtre d'aide
    '''

    message = """
La simulation est asymétrique : chaque population peut avoir des caractéristiques différentes des autres.


Pour chaque population :

    • Vitesse : la distance (par dimension = zone carrée) qu'une entité peut parcourir à chaque étape
    • Dégâts : la quantité maximale d'énergie qu'une entité peut enlever à sa proie
    • Vol d'énergie : la proportion d'énergie qu'une entité récupère après en avoir attaqué une autre (vol * dégats infligés)
    • Energie de naissance : l'énergie d'une entité lors de sa naissance
    • Energie pour reproduction : l'énergie nécessaire à une entité pour se reproduire
    • Facteur de vieillissement : vitesse à laquelle une entité perd naturellement de l'énergie
    • Vision : la distance (par dimension = zone carrée) à laquelle une entité peut voir
    • Portée : la distance (par dimension = zone carrée) à laquelle une entité peut attaquer


Paramètres généraux :

    • Ecart type de modification : écart type de la loi normale utilisée pour la modification du réseau de neurone d'un enfant.
        Plus ce nombre est grand, plus l'enfant sera différent de son parent.
    • Population : le nombre d'individus lors de la création de la simulation


⚠ Les changements sont appliqués au démarrage d'une nouvelle simulation ⚠
"""

    def __init__(self):
        self.showed = False
        self.window = None

    def show(self):
        self.showed = True
        self.window = ctk.CTkToplevel()
        self.window.title('PyRPS - Aide : Paramètres de simulation')
        self.window.geometry('800x600')
        self.window.textbox = ctk.CTkTextbox(master=self.window)
        self.window.textbox.pack(padx=20, pady=20, fill=ctk.BOTH, expand=True)
        # ? TODO : use monospace font ?
        self.window.textbox.insert('0.0', text=HelpWindowManager.message)
        self.window.textbox.configure(state=ctk.DISABLED)
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.window.grab_set()

    def on_close(self):
        self.showed = False
        self.window.destroy()


class FrameTitle(ctk.CTkLabel):
    '''
    Classe pour les titres de section avec des paramètres prédéfinis
    '''
    def __init__(self, master, text):
        super().__init__(master=master, text=text, font=('Arial', 16))


class SimulationControl(ctk.CTkFrame):
    '''
    Classe pour gérer les boutons de contrôle de la simulation
    '''
    def __init__(self, master):
        super().__init__(master=master, fg_color=master.cget('fg_color'), border_width=2)

        self.grid_anchor('center')

        FrameTitle(master=self, text='Simulation').grid(column=0, row=0, columnspan=2, pady=10)

        self.btn_run = ctk.CTkButton(master=self, text='Lancer la simulation', bg_color=self.cget('fg_color'))
        self.btn_run.grid(column=0, row=1, padx=6, pady=6)
        self.btn_stop = ctk.CTkButton(master=self, text='Stopper la simulation', bg_color=self.cget('fg_color'))
        self.btn_stop.grid(column=1, row=1, padx=6, pady=6)
        self.btn_step = ctk.CTkButton(master=self, text='Avancer d\'une étape', bg_color=self.cget('fg_color'))
        self.btn_step.grid(column=0, row=2, padx=6, pady=6)
        self.btn_reset = ctk.CTkButton(master=self, text='Réinitialiser', bg_color=self.cget('fg_color'))
        self.btn_reset.grid(column=1, row=2, padx=6, pady=6)
        self.btn_show_plot = ctk.CTkButton(master=self, text='Afficher le graphe', bg_color=self.cget('fg_color'))
        self.btn_show_plot.grid(column=0, row=3, columnspan=2, padx=6, pady=(6, 20), stick='ew')

    def on_run(self, fn):
        self.btn_run.configure(command=fn)

    def on_stop(self, fn):
        self.btn_stop.configure(command=fn)

    def on_reset(self, fn):
        self.btn_reset.configure(command=fn)

    def on_step(self, fn):
        self.btn_step.configure(command=fn)
    
    def on_show_plot(self, fn):
        self.btn_show_plot.configure(command=fn)


class PopulationAttributesSettings(ctk.CTkFrame):
    '''
    Classe pour la modification des attributs d'une entité
    '''
    def __init__(self, master):
        super().__init__(master=master, fg_color=master.cget('fg_color'))

        ctk.CTkLabel(self, text="Vitesse").grid(row=0, column=0)
        self.speed_var = ctk.IntVar()
        ctk.CTkSlider(self, from_=1, to=10, variable=self.speed_var, number_of_steps=9).grid(row=0, column=1)
        ctk.CTkLabel(self, textvariable=self.speed_var, width=40).grid(row=0, column=2)

        ctk.CTkLabel(self, text="Dégats").grid(row=1, column=0)
        self.damage_var = ctk.IntVar()
        ctk.CTkSlider(self, from_=1, to=20, variable=self.damage_var, number_of_steps=19).grid(row=1, column=1)
        ctk.CTkLabel(self, textvariable=self.damage_var, width=40).grid(row=1, column=2)

        ctk.CTkLabel(self, text="Vol d'énergie").grid(row=2, column=0)
        self.steal_var = ctk.DoubleVar()
        ctk.CTkSlider(self, from_=0, to=1, variable=self.steal_var).grid(row=2, column=1)
        ctk.CTkLabel(self, textvariable=self.steal_var, width=40).grid(row=2, column=2)

        ctk.CTkLabel(self, text="Énergie de naissance").grid(row=3, column=0)
        self.energy_def_var = ctk.IntVar()
        ctk.CTkSlider(self, from_=1, to=200, variable=self.energy_def_var, number_of_steps=199).grid(row=3, column=1)
        ctk.CTkLabel(self, textvariable=self.energy_def_var, width=40).grid(row=3, column=2)

        ctk.CTkLabel(self, text="Énergie pour reproduction").grid(row=4, column=0)
        self.energy_child_var = ctk.IntVar()
        ctk.CTkSlider(self, from_=1, to=200, variable=self.energy_child_var, number_of_steps=199).grid(row=4, column=1)
        ctk.CTkLabel(self, textvariable=self.energy_child_var, width=40).grid(row=4, column=2)

        ctk.CTkLabel(self, text="Facteur de vieillissement").grid(row=5, column=0) # loss_factor
        self.aging_var = ctk.DoubleVar()
        ctk.CTkSlider(self, from_=0, to=1, variable=self.aging_var).grid(row=5, column=1)
        ctk.CTkLabel(self, textvariable=self.aging_var, width=40).grid(row=5, column=2)

        ctk.CTkLabel(self, text="Vision").grid(row=6, column=0)
        self.vision_var = ctk.IntVar()
        ctk.CTkSlider(self, from_=1, to=20, variable=self.vision_var, number_of_steps=19).grid(row=6, column=1)
        ctk.CTkLabel(self, textvariable=self.vision_var, width=40).grid(row=6, column=2)

        ctk.CTkLabel(self, text="Portée").grid(row=7, column=0)
        self.range_var = ctk.IntVar()
        ctk.CTkSlider(self, from_=1, to=20, variable=self.range_var, number_of_steps=19).grid(row=7, column=1)
        ctk.CTkLabel(self, textvariable=self.range_var, width=40).grid(row=7, column=2)

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


class SimulationSettings(ctk.CTkFrame):
    '''
    Classe pour les paramètres des populations et de la simulation
    '''
    def __init__(self, master, params_pointer):
        super().__init__(master=master, fg_color=master.cget('fg_color'), border_width=2)
        self.params = params_pointer

        self.help_window = HelpWindowManager()

        FrameTitle(master=self, text="Paramètres").grid(row=0, column=0, pady=10)

        ctk.CTkButton(self, text='Aide', command=self.on_help).grid(row=1, column=0, pady=6)

        self.actions_frame = ctk.CTkFrame(self)
        ctk.CTkButton(self.actions_frame, text="Reset", command=self.reset_params).grid(row=0, column=0, padx=6, pady=6)
        ctk.CTkButton(self.actions_frame, text="Sauver", command=self.save_params).grid(row=0, column=1, padx=6, pady=6)
        self.actions_frame.grid(row=2, column=0, pady=6)

        self.params_select = ctk.CTkTabview(self, bg_color=self.cget('fg_color'), width=420, height=250) # TODO: trouver la bonne valeur pour height
        self.params_select.grid(row=3, column=0, padx=12, pady=(6, 12))

        self.create_general_settings(self.params_select.add("Général"))

        self.rock_settings = PopulationAttributesSettings(self.params_select.add("Pierre"))
        self.paper_settings = PopulationAttributesSettings(self.params_select.add("Feuille"))
        self.sissors_settings = PopulationAttributesSettings(self.params_select.add("Ciseaux"))
        self.rock_settings.pack()
        self.paper_settings.pack()
        self.sissors_settings.pack()

    def create_general_settings(self, master):
        frame = ctk.CTkFrame(master, fg_color=master.cget('fg_color'))
        frame.pack()

        ctk.CTkLabel(frame, text="Écart type de modification").grid(row=0, column=0, padx=(0, 6))
        self.mod_scale_var = ctk.DoubleVar()
        ctk.CTkSlider(frame, from_=0, to=1, variable=self.mod_scale_var).grid(row=0, column=1)
        ctk.CTkLabel(frame, textvariable=self.mod_scale_var, width=40).grid(row=0, column=2)

        ctk.CTkLabel(frame, text="Population").grid(row=1, column=0)
        self.pop_size_var = ctk.IntVar()
        ctk.CTkSlider(frame, from_=0, to=200, variable=self.pop_size_var).grid(row=1, column=1)
        ctk.CTkLabel(frame, textvariable=self.pop_size_var, width=40).grid(row=1, column=2)

    def on_help(self):
        self.help_window.show()

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
        self.mod_scale_var.set(self.params['sim']['data']['mod_scale'])
        self.pop_size_var.set(self.params['sim']['pop_size'])

    def get_data(self):
        rock = self.rock_settings.get_values()
        paper = self.paper_settings.get_values()
        sissors = self.sissors_settings.get_values()
        data = {k: [paper[k], rock[k], sissors[k]] for k in paper.keys() & rock.keys() & sissors.keys()}
        data['mod_scale'] = self.mod_scale_var.get()
        return {
            "easter_egg": True, # TODO: easter egg from config.json
            "sim": {
                "delta_time": self.params['sim']['delta_time'],
                "grid_size": self.params['sim']['grid_size'],
                "tile_size": self.params['sim']['tile_size'],
                "pop_size": self.pop_size_var.get(),
                "layers": self.params['sim']['layers'],
                "data": data
            }
        }

    def save_params(self):
        self.params = self.get_data()
