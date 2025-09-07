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

# DEFAULTS PATHS
path_APP = os.path.dirname(os.path.abspath(__file__))[:-9]  # Remove o sufixo "\gui_class"
path_save = "analise_resultados/"
path_ArqMain   = "arquivos_principais/"

conv = {
  'temSol': bool,
  'L'     : ast.literal_eval,
  'L_b14' : ast.literal_eval,
  'Llabel': ast.literal_eval,
  'Rlabel': ast.literal_eval
}