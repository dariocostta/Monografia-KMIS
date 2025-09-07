from bibkmis.heuristicaskmis import *
from bibkmis.auxkmis import *
from bibkmis.typeskmis import KMIS

import os
import ast
import pandas as pd
import multiprocessing
import threading
import time
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from enum import IntEnum, Enum, auto

# PATHS
path_APP = os.path.dirname(os.path.abspath(__file__))
path_save = "analise_resultados/"
path_ArqMain   = "arquivos_principais/"

# GLOBAL VARIABLES
MAX_TAMANHO_L : int = 300  # Valor padrão, será atualizado ao carregar as instâncias
DFI : pd.DataFrame = pd.DataFrame()  # DataFrame global para armazenar as instâncias carregadas
DFAT : pd.DataFrame = pd.DataFrame()  # Teste de parametros
DFRT : pd.DataFrame = pd.DataFrame()  # Resultados
DFIRT : pd.DataFrame = pd.DataFrame()  # Resultados nas Instancias reduzidas

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

# ==== DICT TO STORE KMIS INSTANCES ==== temporarily
dictI = {
    "id": [], "kmis": [], "p": [], "k": [],
    "|L|": [], "|R|": [], "L": [], 'temSol': [], 'classe': [],
    "|L|_b14": [], "|R|_b14": [], "L_b14": [], "Llabel_b14": [], "Rlabel_b14": [], "kmis_b14": []
}
conv = {
  'temSol': bool,
  'L'     : ast.literal_eval,
  'L_b14' : ast.literal_eval,
  'Llabel': ast.literal_eval,
  'Rlabel': ast.literal_eval
}

# Load bar indices for the first screen
class LoadBar(IntEnum):
    INSTANCIAS = 0
    TESTE_PARAM = 1
    TESTE_COMPLETO = 2

class LoadStage(IntEnum):
    CSV_INSTANCIAS = 0
    RECREATING_INSTANCIAS = 1
    CSV_TESTE_PARAM = 2
    CSV_RESULT = 3
    CSV_RESULT_REDUZIDAS = 4
    FINAL = 5
    

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("KMIS")
        self.geometry("700x500")
        self.configure(bg="#f5f5f5")
        self.set_style()
        self.create_widgets()

    def set_style(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure("green.Horizontal.TProgressbar", foreground='green', background='green')
        style.configure("TLabel", background="#f5f5f5", font=("Arial", 11))
        style.configure("TFrame", background="#f5f5f5")
        style.configure("TButton", font=("Arial", 10))

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
        form_frame.pack(pady=10)

        self.param_vars = {}

        # Array of sizes (comma-separated)
        ttk.Label(form_frame, text="Tamanhos das Instâncias:").grid(row=0, column=0, sticky="w", pady=2)
        self.param_vars['sizes'] = tk.StringVar()
        self.param_vars['sizes'].set(defaults["gerar"]["sizes"])
        sizes_entry = ttk.Entry(form_frame, textvariable=self.param_vars['sizes'])
        sizes_entry.grid(row=0, column=1, pady=2)

        # Number of instances per class (default slider, stable label)
        ttk.Label(form_frame, text="Instâncias por Classe:").grid(row=1, column=0, sticky="w", pady=2)
        self.param_vars['num_per_class'] = tk.IntVar()
        self.param_vars['num_per_class'].set(defaults["gerar"]["num_per_class"])
        num_scale = ttk.Scale(
            form_frame, from_=1, to=10, orient="horizontal",
            variable=self.param_vars['num_per_class'], length=90,
            command=lambda v: self.param_vars['num_per_class'].set(int(float(v)))
        )
        num_scale.grid(row=1, column=1, pady=2, sticky="w")
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

class StartupWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Carregando Dados KMIS")
        self.geometry("400x220")
        self.resizable(False, False)
        self.configure(bg="#f5f5f5")
        self.set_style()

        self.labels = ["Instâncias", "Teste de Parâmetros", "Resultados"]
        self.progress_bars = []
        self.percent_labels = []

        for i, label in enumerate(self.labels):
            # Label alinhada à esquerda
            ttk.Label( self, text=label, style="TLabel", anchor="w").pack( fill="x", anchor="w", padx=20,  pady=(15 if i==0 else 5, 0))

            # Frame que agrupa progress bar + %  
            frame = ttk.Frame(self, style="TFrame")
            frame.pack( fill="x", anchor="w", padx=20, pady=2)
            pb = ttk.Progressbar( frame, length=300, mode='determinate', maximum=100, style="green.Horizontal.TProgressbar")
            pb.pack(side="left")
            percent = ttk.Label( frame, text="0%", width=5, anchor="e", style="TLabel")
            percent.pack(side="left", padx=8)
            self.progress_bars.append(pb)
            self.percent_labels.append(percent)

        self.ok_btn = ttk.Button(self, text="OK", command=self.launch_main, state="disabled", style="TButton")
        self.ok_btn.pack(pady=15)

        self.start_loading()

    def set_style(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure("green.Horizontal.TProgressbar", foreground='green', background='green')
        style.configure("red.Horizontal.TProgressbar", foreground='red', background='red')
        style.configure("gray.Horizontal.TProgressbar", foreground='black', background='gray')
        style.configure("TLabel", background="#f5f5f5", font=("Arial", 11))
        style.configure("TFrame", background="#f5f5f5")
        style.configure("TButton", font=("Arial", 10))

    def start_loading(self, i=0, stage=LoadStage.CSV_INSTANCIAS):
        global DFI, DFAT, DFRT, DFIRT, dictI, MAX_TAMANHO_L, conv
        if stage == LoadStage.CSV_INSTANCIAS:
            # ==== Carregamento de Instâncias Salvas =====
            try:
                DFI = pd.read_csv(path_APP+"/"+path_ArqMain+"instancias.csv", converters=conv)
                print(f'Leitura de instancias.csv ({DFI.shape[0]} linhas) bem sucedida.')
            except:
                print(f'ERRO NO ARQUIVO instancias.csv')
                self.progress_bars[LoadBar.INSTANCIAS].config(style="gray.Horizontal.TProgressbar", value=100)
                DFI = pd.DataFrame(columns = list(dictI.keys()))
            finally:
                if DFI.shape[0] == 0:
                    self.after(1, self.start_loading, 0, LoadStage.CSV_TESTE_PARAM)
                else:
                    self.after(1, self.start_loading, 0, LoadStage.RECREATING_INSTANCIAS)

        if stage == LoadStage.RECREATING_INSTANCIAS: # Slow, so threaded
            # ==== Recriação dos objetos KMIS a partir do DataFrame DFI =====
            worker = threading.Thread(
                target=self._recreate_instances_thread,
                daemon=True
            )
            worker.start()

        if stage == LoadStage.CSV_TESTE_PARAM:
            try:
                # ==== Carregamento de Teste de parametros =====
                DFAT = pd.read_csv(path_APP+"/"+path_ArqMain+"teste_parametros.csv", converters=conv)
                print(f'Leitura de teste_parametros.csv ({DFAT.shape[0]} linhas) bem sucedida.')
                self.progress_bars[LoadBar.TESTE_PARAM].config(value=100)
                self.percent_labels[LoadBar.TESTE_PARAM]['text'] = "100%"
                self.update_idletasks()
            except:
                self.progress_bars[LoadBar.TESTE_PARAM].config(style="gray.Horizontal.TProgressbar", value=100)
                print(f'ERRO NO ARQUIVO teste_parametros.csv')
                DFAT = pd.DataFrame()
            finally:
                self.update_idletasks()
                self.after(1, self.start_loading, 0, LoadStage.CSV_RESULT)

        if stage == LoadStage.CSV_RESULT:
            try:
                DFRT = pd.read_csv(path_APP+"/"+path_ArqMain+"resultados.csv", converters=conv)
                print(f'Leitura de resultados.csv ({DFRT.shape[0]} linhas) bem sucedida.')
                self.progress_bars[LoadBar.TESTE_COMPLETO].config(value=50)
                self.percent_labels[LoadBar.TESTE_COMPLETO]['text'] = "50%"
                self.update_idletasks()
            except:
                self.progress_bars[LoadBar.TESTE_COMPLETO].config(style="gray.Horizontal.TProgressbar", value=50)
                print(f'ERRO NO ARQUIVO resultados.csv')
                DFRT = pd.DataFrame()
                self.update_idletasks()
            finally:
                self.update_idletasks()
                self.after(200, self.start_loading, 0, LoadStage.CSV_RESULT_REDUZIDAS)
                
        if stage == LoadStage.CSV_RESULT_REDUZIDAS:
            try:
                DFIRT = pd.read_csv(path_APP+"/"+path_ArqMain+"resultados_reduzidas.csv", converters=conv)
                print(f'Leitura de resultados_reduzidas.csv ({DFIRT.shape[0]} linhas) bem sucedida.')
                self.progress_bars[LoadBar.TESTE_COMPLETO].config(value=100)
                self.percent_labels[LoadBar.TESTE_COMPLETO]['text'] = "100%"
                self.update_idletasks()
            except:
                self.progress_bars[LoadBar.TESTE_COMPLETO].config(style="gray.Horizontal.TProgressbar", value=100)
                print(f'ERRO NO ARQUIVO resultados_reduzidas.csv')
                DFIRT = pd.DataFrame()
                self.update_idletasks()
            finally:
                self.update_idletasks()
                self.after(1, self.start_loading, 0, LoadStage.FINAL)

        if stage == LoadStage.FINAL:
            self.ok_btn['state'] = "normal"
            self.update_idletasks()

    def _recreate_instances_thread(self): # Slow, so threaded
        global DFI, dictI, MAX_TAMANHO_L
        total = DFI.shape[0]
        for i in range(total):
            try:
                row = DFI.iloc[i]
                kmis = KMIS(int(row['|L|']), int(row['|R|']), float(row['p']), int(row['k']), row['L'])
                kmis_reduzido = KMIS(int(row['|L|_b14']), int(row['|R|_b14']), float(row['p']), int(row['k']), row['L_b14'])
                kmis_reduzido.Llabel = row['Llabel_b14']
                kmis_reduzido.Rlabel = row['Rlabel_b14']
                dictI['kmis'].append(kmis)
                dictI['kmis_b14'].append(kmis_reduzido)
            except (KeyError, TypeError, ValueError) as e:
                self.after(5, lambda : self.progress_bars[LoadBar.INSTANCIAS].config(style="red.Horizontal.TProgressbar", value=100))
                print(f"Erro ao processar linha {i}: {e}")
            finally:
                pct  = int((i+1) / total * 100)
                text = f"{(i+1) / total * 100:.1f}%"
                self.after(16, lambda pct=pct, text=text: (
                    self.progress_bars[LoadBar.INSTANCIAS].config(value=pct),
                    self.percent_labels[LoadBar.INSTANCIAS].config(text=text),
                    self.update_idletasks()
                ))

        try:
            DFI['kmis'] = dictI['kmis']
            DFI['kmis_b14'] = dictI['kmis_b14']
            tamanhos_L = DFI[DFI['temSol']]['|L|'].value_counts().reset_index().sort_values(by='|L|')
            MAX_TAMANHO_L = int(tamanhos_L['|L|'].max())
        except Exception as e:
            self.after(16, lambda: (
                self.progress_bars[LoadBar.INSTANCIAS].config(style="red.Horizontal.TProgressbar", value=100),
                messagebox.showerror("Erro", f"Erro ao coletar parâmetros: {e}")
            ))
        finally:
            self.after(16, lambda: (
                self.progress_bars[LoadBar.INSTANCIAS].config(value=100),
                self.percent_labels[LoadBar.INSTANCIAS].config(text="100%"),
                self.start_loading(0, LoadStage.CSV_TESTE_PARAM)
            ))

    def launch_main(self):
        self.destroy()
        app = MainWindow()
        app.mainloop()

if __name__ == "__main__":
    StartupWindow().mainloop()