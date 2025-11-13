from gui_class.gui_bib import *

# ==== DEFAULT VALUES DICT ====
defaults = {
    "gerar": {
        "sizes": "40, 60, 80, 100, 140, 180, 200, 240, 280, 300",
        "num_per_class": 2
    },
    "teste": {
        "tamGrupoTreino": 34,
        "numRep_A": 10,
        "t_lim_A": 10,
        "tempo_save_A": 300,
        "limite_agendamentos": 40,
        "nucleos_teste": min(10, max(2, multiprocessing.cpu_count())-1)
    }
}

class MainWindow(tk.Tk):
    def __init__(self, globalInfo):
        super().__init__()
        self.G = globalInfo
        self.title("KMIS")
        self.geometry("800x600")
        style = ttk.Style(self)
        style.theme_use('alt')
        style.configure("green.Horizontal.TProgressbar", foreground='green', background='green')
        self.create_widgets()
        print(self.G.DFI.shape[0], " instâncias carregadas.")


    def reset_defaults(self, caller : str):
        if caller == "Teste de Parâmetros":
            for var in defaults["teste"]:
                self.teste_vars[var].set(defaults["teste"][var])
        elif caller == "Gerar Instâncias":
            for var in defaults["gerar"]:
                self.param_vars[var].set(defaults["gerar"][var])

    def create_widgets(self):
        # Top bar with tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)

        # First tab: Gerar Instâncias
        self.frame_gerar = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_gerar, text="Gerar Instâncias")
        self.create_gerar_widgets(self.frame_gerar)

        # Second tab: Teste de Parâmetros
        self.frame_teste = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_teste, text="Teste de Parâmetros")
        self.create_teste_widgets(self.frame_teste)

    def create_gerar_widgets(self, parent):
        ttk.Label(parent, text="Gerar Instâncias", font=("Arial", 14)).pack(pady=10)


        form_frame = ttk.Frame(parent)
        form_frame.columnconfigure(2, weight=1)
        form_frame.pack(pady=10)

        self.param_vars = {}

        # Array of sizes (comma-separated)
        ttk.Label(form_frame, text="Tamanhos das Instâncias:").grid(row=0, column=0, sticky="w", pady=2)
        self.param_vars['sizes'] = tk.StringVar()
        self.param_vars['sizes'].set(defaults["gerar"]["sizes"])
        sizes_entry = ttk.Entry(form_frame, textvariable = self.param_vars['sizes'], width=50)
        sizes_entry.grid(row=0, column=1, pady=2)

        # Number of instances per class (default slider, stable label)
        ttk.Label(form_frame, text="Instâncias por Classe:").grid(row=1, column=0, sticky="w", pady=2)
        self.param_vars['num_per_class'] = tk.IntVar()
        self.param_vars['num_per_class'].set(defaults["gerar"]["num_per_class"])
        num_scale = ttk.Scale(
            form_frame, from_=1, to=10, orient="horizontal",
            variable=self.param_vars['num_per_class'], length=100,
            command=lambda v: self.param_vars['num_per_class'].set(int(float(v)))
        )
        num_scale.grid(row=1, column=1, pady=2, sticky="ew")
        # Stable value label with fixed width, vertically centered
        self.num_label = ttk.Label(form_frame, textvariable=self.param_vars['num_per_class'], width=2, anchor="center", font=("Arial", 10))
        self.num_label.grid(row=1, column=2, padx=5, sticky="n")

        # Generate button
        self.generate_btn = ttk.Button(parent, text="Gerar Instâncias", command=self.on_generate)
        self.generate_btn.pack(pady=10)

        # Reset button
        ttk.Button(parent, text="Resetar Valores", command=lambda: self.reset_defaults("Gerar Instâncias")).pack(pady=5)


    def create_teste_widgets(self, parent):
        ttk.Label(parent, text="Teste de Parâmetros", font=("Arial", 14)).pack(pady=10)
        form_frame = ttk.Frame(parent)
        form_frame.pack(pady=10)
        self.teste_vars = {}

        # Configuração dos campos: (label, var_name, var_type, from_, to, width)
        campos = [
            ("Tamanho Grupo Treino:", 'tamGrupoTreino', tk.IntVar, 1, 50, 5),
            ("Num. Rep. A:", 'numRep_A', tk.IntVar, 2, 20, 5),
            ("Tempo Limite A (s):", 't_lim_A', tk.DoubleVar, 0.5, 30, 5),
            ("Tempo Save A (s):", 'tempo_save_A', tk.IntVar, 5, 600, 5),
            ("Limite Agendamentos:", 'limite_agendamentos', tk.IntVar, 1, 50, 5),
            ("Núcleos Teste:", 'nucleos_teste', tk.IntVar, 1, max(2, multiprocessing.cpu_count())-1, 5),
        ]

        for i, (label, varname, vartype, vmin, vmax, width) in enumerate(campos):
            ttk.Label(form_frame, text=label).grid(row=i, column=0, sticky="w", pady=2)
            if varname not in self.teste_vars:
                default = defaults["teste"].get(varname, vmin)
                self.teste_vars[varname] = vartype(value=default)
            # Ajuste para step e callback especial
            if varname == 't_lim_A':
                def update(v, vn=varname):
                    val = round(float(v) * 2) / 2
                    self.teste_vars[vn].set(val)
                cmd = update
            elif varname == 'tempo_save_A':
                def update(v, vn=varname):
                    val = int(round(float(v) / 5) * 5)
                    self.teste_vars[vn].set(val)
                cmd = update
            else:
                cmd = lambda v, vn=varname: self.teste_vars[vn].set(int(float(v)))
            ttk.Scale(
                form_frame, from_=vmin, to=vmax, orient="horizontal", length=150,
                variable=self.teste_vars[varname], command=cmd
            ).grid(row=i, column=1, pady=2, sticky="w")
            ttk.Label(
                form_frame, textvariable=self.teste_vars[varname],
                width=width, anchor="e", justify="right"
            ).grid(row=i, column=2, padx=(0,8))

        ttk.Button(parent, text="Executar Teste de Parâmetros", command=self.on_teste_parametros).pack(pady=10)
        ttk.Button(parent, text="Resetar Valores", command=lambda: self.reset_defaults("Teste de Parâmetros")).pack(pady=5)


    def on_generate(self):
        # Collect parameters with type validation
        try:
            sizes_str = self.param_vars['sizes'].get()
            sizes = [int(s.strip()) for s in sizes_str.split(',') if s.strip()]
            if not sizes:
                raise ValueError
            num_per_class = self.param_vars['num_per_class'].get()
            params = {
                'sizes': sizes,
                'num_per_class': num_per_class
            }
        except ValueError:
            messagebox.showerror("Erro", "Por favor, insira tamanhos válidos (ex: 10,20,30) e um número de instâncias por classe entre 1 e 10.")
            return
        messagebox.showinfo("Parâmetros coletados", str(params))
        # Here you would call the instance generation logic with these params

    def on_teste_parametros(self):
        # Coleta e mostra os parâmetros do teste de parâmetros
        try:
            params = {
                'tamGrupoTreino': self.teste_vars['tamGrupoTreino'].get(),
                'numRep_A': self.teste_vars['numRep_A'].get(),
                't_lim_A': self.teste_vars['t_lim_A'].get(),
                'tempo_save_A': self.teste_vars['tempo_save_A'].get(),
                'limite_agendamentos': self.teste_vars['limite_agendamentos'].get(),
                'nucleos_teste': self.teste_vars['nucleos_teste'].get()
            }
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao coletar parâmetros: {e}")
            return
        messagebox.showinfo("Parâmetros Teste de Parâmetros", str(params))
        # Aqui você pode chamar a lógica de teste de parâmetros com esses params