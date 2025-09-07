from gui_class.gui_bib import *
from gui_class.StartupWindow import StartupWindow
from gui_class.MainWindow import MainWindow

class KMISApp: 	# Singleton to hold global variables
    MAX_TAMANHO_L : int = 300  # Valor padrão, será atualizado ao carregar as instâncias
    DFI : pd.DataFrame = pd.DataFrame()    # DataFrame global para armazenar as instâncias carregadas
    DFAT : pd.DataFrame = pd.DataFrame()   # Teste de parametros
    DFRT : pd.DataFrame = pd.DataFrame()   # Resultados
    DFIRT : pd.DataFrame = pd.DataFrame()  # Resultados nas Instancias reduzidas
    # ==== DICT TO STORE KMIS INSTANCES ==== temporarily
    dictI = {
    	"id": [], "kmis": [], "p": [], "k": [],
    	"|L|": [], "|R|": [], "L": [], 'temSol': [], 'classe': [],
    	"|L|_b14": [], "|R|_b14": [], "L_b14": [], "Llabel_b14": [], "Rlabel_b14": [], "kmis_b14": []
    }

if __name__ == "__main__":
    globalKMIS = KMISApp()
    StartupWindow(globalKMIS).mainloop()
    MainWindow(globalKMIS).mainloop()