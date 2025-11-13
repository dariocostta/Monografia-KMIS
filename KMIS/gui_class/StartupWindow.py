from gui_class.gui_bib import *

# Load bar indices for the first screen
class LoadBar(IntEnum):
    INSTANCIAS = 0
    TESTE_PARAM = 1
    TESTE_COMPLETO = 2

# Stages of loading process
class LoadStage(IntEnum):
    CSV_INSTANCIAS = 0
    RECREATING_INSTANCIAS = 1
    CSV_TESTE_PARAM = 2
    CSV_RESULT = 3
    CSV_RESULT_REDUZIDAS = 4
    FINAL = 5

class StartupWindow(tk.Tk):
    def __init__(self, globalInfo):
        super().__init__()
        self.G = globalInfo
        self.configure(bg="white")
        main_frame = ttk.Frame(self, style="White.TFrame")

        style = ttk.Style(self)
        style.theme_use('alt')
        style.configure("TFrame", background="white")
        style.configure("TLabel", background="white")
        style.configure("green.Horizontal.TProgressbar", foreground="green", background="green")
        style.configure("red.Horizontal.TProgressbar", foreground="red", background="red")
        style.configure("gray.Horizontal.TProgressbar", foreground="black", background="gray")

        self.title("Carregando Dados KMIS")
        self.geometry("800x600")
        # Divide root into three rows to vertically center main_frame
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)

        # Container for all widgets, centered in root
        main_frame = ttk.Frame(self)
        main_frame.grid(row=1, column=0)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=0)

        self.labels = ["Instâncias", "Teste de Parâmetros", "Resultados"]
        self.progress_bars = []
        self.percent_labels = []

        for i, text in enumerate(self.labels):
            row_label = i * 2
            row_bar = row_label + 1

            # Section label
            lbl = ttk.Label( main_frame, text=text, style="TLabel", anchor="w")
            lbl.grid( row=row_label, column=0, columnspan=2, sticky="w", padx=20, pady=(15 if i == 0 else 5, 0))

            # Progress bar
            pb = ttk.Progressbar( main_frame, length=300, mode="determinate", maximum=100, style="green.Horizontal.TProgressbar")
            pb.grid( row=row_bar, column=0, sticky="w", padx=(20, 0), pady=2)

            # Percentage label
            percent = ttk.Label( main_frame, text="0%", width=6, anchor="e", style="TLabel")
            percent.grid( row=row_bar, column=1, sticky="e", padx=(5, 20))

            self.progress_bars.append(pb)
            self.percent_labels.append(percent)

        # OK button centered under the progress bars
        btn_row = len(self.labels) * 2
        self.ok_btn = ttk.Button( main_frame, text="OK", command=self.destroy, state="disabled", style="TButton")
        self.ok_btn.grid( row=btn_row, column=0, columnspan=2, pady=15)

        self.start_loading()

    def start_loading(self, stage=LoadStage.CSV_INSTANCIAS):
        # Stage: carregar instâncias (CSV_INSTANCIAS)
        if stage == LoadStage.CSV_INSTANCIAS:
            try:
                self.G.DFI = pd.read_csv(
                    f"{path_APP}/{path_ArqMain}instancias.csv",
                    converters=conv
                )
                print(f"Leitura de instancias.csv ({self.G.DFI.shape[0]} linhas) bem sucedida.")
            except:
                print("ERRO NO ARQUIVO instancias.csv")
                self.progress_bars[LoadBar.INSTANCIAS].config(style="gray.Horizontal.TProgressbar", value=100)
                self.G.DFI = pd.DataFrame(columns=list(self.G.dictI.keys()))
            finally:
                next_stage = LoadStage.CSV_TESTE_PARAM if self.G.DFI.shape[0] == 0 \
                             else LoadStage.RECREATING_INSTANCIAS
                self.after(1, self.start_loading, next_stage)

        # Stage: recriar instâncias (RECREATING_INSTANCIAS)
        if stage == LoadStage.RECREATING_INSTANCIAS:
            worker = threading.Thread(
                target=self._recreate_instances_thread,
                daemon=True
            )
            worker.start()

        # Stage: carregar teste de parâmetros (CSV_TESTE_PARAM)
        if stage == LoadStage.CSV_TESTE_PARAM:
            try:
                self.G.DFAT = pd.read_csv(
                    f"{path_APP}/{path_ArqMain}teste_parametros.csv",
                    converters=conv
                )
                print(f"Leitura de teste_parametros.csv ({self.G.DFAT.shape[0]} linhas) bem sucedida.")
                self.progress_bars[LoadBar.TESTE_PARAM].config(value=100)
                self.percent_labels[LoadBar.TESTE_PARAM]['text'] = "100%"
                self.update_idletasks()
            except:
                print("ERRO NO ARQUIVO teste_parametros.csv")
                self.progress_bars[LoadBar.TESTE_PARAM] \
                    .config(style="gray.Horizontal.TProgressbar", value=100)
                self.G.DFAT = pd.DataFrame()
            finally:
                self.update_idletasks()
                self.after(1, self.start_loading, LoadStage.CSV_RESULT)

        # Stage: carregar resultados (CSV_RESULT)
        if stage == LoadStage.CSV_RESULT:
            try:
                self.G.DFRT = pd.read_csv(
                    f"{path_APP}/{path_ArqMain}resultados.csv",
                    converters=conv
                )
                print(f"Leitura de resultados.csv ({self.G.DFRT.shape[0]} linhas) bem sucedida.")
                self.progress_bars[LoadBar.TESTE_COMPLETO].config(value=50)
                self.percent_labels[LoadBar.TESTE_COMPLETO]['text'] = "50%"
                self.update_idletasks()
            except:
                print("ERRO NO ARQUIVO resultados.csv")
                self.progress_bars[LoadBar.TESTE_COMPLETO] \
                    .config(style="gray.Horizontal.TProgressbar", value=50)
                self.G.DFRT = pd.DataFrame()
                self.update_idletasks()
            finally:
                self.update_idletasks()
                self.after(200, self.start_loading, LoadStage.CSV_RESULT_REDUZIDAS)

        # Stage: carregar resultados reduzidas (CSV_RESULT_REDUZIDAS)
        if stage == LoadStage.CSV_RESULT_REDUZIDAS:
            try:
                self.G.DFIRT = pd.read_csv(
                    f"{path_APP}/{path_ArqMain}resultados_reduzidas.csv",
                    converters=conv
                )
                print(f"Leitura de resultados_reduzidas.csv ({self.G.DFIRT.shape[0]} linhas) bem sucedida.")
                self.progress_bars[LoadBar.TESTE_COMPLETO].config(value=100)
                self.percent_labels[LoadBar.TESTE_COMPLETO]['text'] = "100%"
                self.update_idletasks()
            except:
                print("ERRO NO ARQUIVO resultados_reduzidas.csv")
                self.progress_bars[LoadBar.TESTE_COMPLETO].config(style="gray.Horizontal.TProgressbar", value=100)
                self.G.DFIRT = pd.DataFrame()
                self.update_idletasks()
            finally:
                self.after(1, self.start_loading, LoadStage.FINAL)

        # Stage final: habilita botão OK
        if stage == LoadStage.FINAL:
            self.ok_btn['state'] = "normal"
            self.update_idletasks()

    def _recreate_instances_thread(self):
        total = self.G.DFI.shape[0]
        for i in range(total):
            try:
                row = self.G.DFI.iloc[i]
                kmis = KMIS(
                    int(row['|L|']), int(row['|R|']),
                    float(row['p']), int(row['k']),
                    row['L']
                )
                kmis_reduzido = KMIS(
                    int(row['|L|_b14']), int(row['|R|_b14']),
                    float(row['p']), int(row['k']),
                    row['L_b14']
                )
                kmis_reduzido.Llabel = row['Llabel_b14']
                kmis_reduzido.Rlabel = row['Rlabel_b14']
                self.G.dictI['kmis'].append(kmis)
                self.G.dictI['kmis_b14'].append(kmis_reduzido)
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
                self.after(5, lambda:
                    self.progress_bars[LoadBar.INSTANCIAS].config(style="red.Horizontal.TProgressbar")
                )
                print(f"Erro ao processar linha {i}: {e}")
            finally:
                pct = int((i + 1) / total * 100)
                text = f"{(i + 1) / total * 100:.1f}%"
                self.after(16, lambda pct=pct, text=text: (
                    self.progress_bars[LoadBar.INSTANCIAS].config(value=pct),
                    self.percent_labels[LoadBar.INSTANCIAS].config(text=text),
                    self.update_idletasks()
                ))

        try:
            self.G.DFI['kmis'] = self.G.dictI['kmis']
            self.G.DFI['kmis_b14'] = self.G.dictI['kmis_b14']

            tamanhos_L = self.G.DFI[self.G.DFI['temSol']]['|L|'].value_counts().reset_index().sort_values(by='|L|')
            self.G.MAX_TAMANHO_L = int(tamanhos_L['|L|'].max())
        except Exception as e:
            self.after(16, lambda: (
                self.progress_bars[LoadBar.INSTANCIAS].config(style="red.Horizontal.TProgressbar", value=100),
                messagebox.showerror("Erro", f"Erro ao coletar parâmetros: {e}")
            ))
        finally:
            self.after(16, lambda: (
                self.progress_bars[LoadBar.INSTANCIAS].config(value=100),
                self.percent_labels[LoadBar.INSTANCIAS].config(text="100%"),
                self.start_loading(LoadStage.CSV_TESTE_PARAM)
            ))