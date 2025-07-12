import networkx as nx           #Plote de Grafo Bipartido
from bibkmis.typeskmis import *
from bibkmis.heuristicaskmis import *
import pandas as pd             # DataFrames
import matplotlib.pyplot as plt
import os                       # Controle de pastas

INTERACTIONS = {} #Entradas do usuario pela get_bool

Heuristicas = {
    'KIEst'       : kInterEstendida,
    # 'HG'          : HG_Bogue13,
    # 'VND'         : VND,
    # 'LS'          : LocalSearch,
    # 'TS'          : TabuSearch,
    'GRASP_RG_TS' : GRASP_RG_TS,
    'GRASP_RG_VND': GRASP_RG_VND,
    'GRASP_RG_VND2': GRASP_RG_VND2,
    'ANT_TS'      : ANT_TS,
    'ANT_VND'     : ANT_VND,
    'ANT_VND2'    : ANT_VND2,
    'ANT2_TS'     : ANT2_TS,
    'ANT2_VND'    : ANT2_VND,
    'ANT2_VND2'   : ANT2_VND2,
}

#============= Todas as funções secundarias usadas para rodar os testes ==============
def run(kmis, H, arg) -> tuple[int, float, SOLUCAO]:
  t_inicio = time.time()
  Solucao : SOLUCAO = H(kmis, **arg)
  t_gasto = time.time() - t_inicio
  Valor : int = kmis.intersect(Solucao)
  assert len(set(Solucao.L_linha)) == kmis.k, "Bug encontrado! Subconjuntos repetidos ou faltantes!"
  return Valor, t_gasto, Solucao

def run_wrapper(idH : str, argID : str, kmis : KMIS, arg : dict, k : int, rep : int
) -> tuple[int, float, SOLUCAO, str, str, int, int]:
    val, tempo, sol = run(kmis, Heuristicas[idH], arg)
    return val, tempo, sol, idH, argID, k, rep

def save_df(dictVector : dict, number : int, pathtype : str) -> None:
  dfSave = pd.DataFrame(dictVector)
  dfSave.to_csv(f'saves_{pathtype}/{pathtype}_{number}.csv', index=False)


#=================       Criação de instancias    ==============================
def geraKMIS(tamL : int = 10, tamR : int = 10,  p : float = 0.4, k : int = 3) -> KMIS:
  """Cria a instância do K-MIS, sorteando as |L|.|R|.p arestas."""

  kmis = KMIS(tamL, tamR, p, k)
  num_e = int(p*(tamL*tamR) + 0.5) #inteiro mais proximo
  E_completo = [(u, v) for u in range(tamL) for v in range(tamR)]
  E_escolhidos = np.random.permutation(E_completo)[:num_e]

  for u,v in E_escolhidos:
    kmis.L[u]+= (1<<(int(v)))
    kmis.R[v]+= (1<<(int(u)))

  return kmis

#=========== Representação Grafica ==============
def graph(kmis : KMIS, A : list[int] = [], isSol = True) -> None:
  # Grafo bipartido
  G = nx.Graph()
  L = ['$S_{'+f'{Si}'+'}$' for Si in kmis.Llabel]
  R = [i for i in kmis.Rlabel]

  L_c = ['lightgray' for _ in range(kmis.tamL + kmis.tamR)]

  E = []
  E_c = []

  for Si in range(kmis.tamL):
    Si_bin = (bin(kmis.L[Si])[2:]).zfill(kmis.tamR)[::-1]# int -> bin
    for bi in range(kmis.tamR):
      if(Si_bin[bi] == '1'):
        E.append(('$S_{'+f'{kmis.Llabel[Si]}'+'}$', kmis.Rlabel[bi]))
        E_c.append('lightgray')
        if isSol:
          if Si in A: E_c[-1] = 'lightcoral'
        else:
          if bi in A: E_c[-1] = 'blue'

  if(isSol):
    sol = SOLUCAO(kmis.tamL, kmis.k)
    sol.L_linha = A
    sol = kmis.intersect(sol, 1)
    sol = (bin(sol)[2:]).zfill(kmis.tamR)[::-1]
    for i in A:
      L_c[i] = 'coral'
    for i in range(kmis.tamR):
      if(sol[i] == '1'):
        L_c[kmis.tamL + i] = 'lightgreen'
  else:
    Nvks = A[0]
    for ki in A[1:]:
      Nvks = Nvks & kmis.R[ki]
    for i in A:
      L_c[kmis.tamL+i] = 'lightblue'

    str_BIN = (bin(Nvks)[2:]).zfill(kmis.tamL)[::-1]
    for i in range(kmis.tamL):
      if(str_BIN[i] == '1'):
        L_c[i] = 'lightgreen'

  G.add_nodes_from(L, bipartite=0)
  G.add_nodes_from(R, bipartite=1)
  G.add_edges_from(E)

  # Ajustando o tamanho da figura
  plt.figure(figsize=(6, int(max([kmis.tamL, kmis.tamR])*0.3 + 2)))

  # Desenhando o grafo
  pos = nx.bipartite_layout(G, nodes=R, align='vertical', scale=-1)
  nx.draw(G, pos, with_labels=True, node_size=400, node_color=L_c, edge_color=E_c)

  print(f'|L|:{kmis.tamL} |R|:{kmis.tamR} p:{kmis.p:.2} k:{kmis.k}') #\nL:', end=''
  #for i in range(kmis.tamL):
  #  if i>12: break
  #  if i%4==0 and i > 0:
  #    print('\n  ', end='')
  #  print((bin(kmis.L[i])[2:].zfill(kmis.tamR))[::-1], end=' ')

  plt.show()

#==================== Append em dict de vetor =======================
def dictAppend(d : dict, arg : dict) -> None:
  for i in d:
    d[i].append(arg[i])

#================= Teste de id de instancia ========================
def teste_consistencia(dfI, df) -> None:
  ids_base = set(dfI['id'])
  ids   = set(df['idI'])
  inc = { 'id_incompativel': ids - ids_base }
  if not inc['id_incompativel']:
    print("\n\t✓ Todos os ids estão presentes em dfI.")
  else:
    print(f"Atenção: {len(inc['id_incompativel'])} id(s) não encontrados em dfI:", sorted(inc['id_incompativel']))


#===================================================
def get_Classe_e_p(kmis : KMIS) -> tuple[str, float]:
  classes = {
    (0.1, 0.1): 'C1', (0.1, 0.4): 'C2', (0.1, 0.7): 'C3',
    (0.4, 0.1): 'C4', (0.4, 0.4): 'C5', (0.4, 0.7): 'C6',
    (0.7, 0.1): 'C7', (0.7, 0.4): 'C8', (0.7, 0.7): 'C9'
  }
  toClass = lambda x: 0.1 if(x > 0 and x<=0.35) else 0.4 if(x>0.35 and x<=0.65) else 0.7
  maxE = kmis.tamL * kmis.tamR
  countE = 0
  for Si in kmis.L:
    countE+=Si.bit_count()
  p = countE/maxE
  classe = classes[toClass(p), toClass((kmis.k/kmis.tamL))]
  return classe, p

#=====================================================
def get_boolean_input(prompt, name):
  os.system('cls' if os.name == 'nt' else 'clear')
  count_interact = 0
  for interac in INTERACTIONS:
    print(f'| {interac} : {INTERACTIONS[interac]} |', end='')
    count_interact+=1
    if count_interact % 5 == 0 and count_interact > 0:
      print('')
  print('\n')
  while True:
      user_input = input(prompt + "\n\t (S/N): ").strip().lower()
      if user_input in ['s', 'y','yes', 'sim']:
          INTERACTIONS[name[:7]+'_'+str(count_interact+1)] = 'S'
          return True
      elif user_input in ['n', 'no', 'nao', 'não']:
          INTERACTIONS[name[:7]+'_'+str(count_interact+1)] = 'N'
          return False
      else:
          print("Por favor, responda: 's' ou 'n'.")

