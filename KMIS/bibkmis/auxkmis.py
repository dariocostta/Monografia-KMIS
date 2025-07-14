import networkx as nx           #Plote de Grafo Bipartido
from bibkmis.typeskmis import *
from bibkmis.heuristicaskmis import *
import pandas as pd             # DataFrames
import matplotlib.pyplot as plt
import os                       # Controle de pastas

INTERACTIONS = {} #Entradas do usuario pela get_bool

classes = {
    (0.1, 0.1): 'C1', (0.1, 0.4): 'C2', (0.1, 0.7): 'C3',
    (0.4, 0.1): 'C4', (0.4, 0.4): 'C5', (0.4, 0.7): 'C6',
    (0.7, 0.1): 'C7', (0.7, 0.4): 'C8', (0.7, 0.7): 'C9'
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


"""**Conjuntos**"""

def Lu(kmis : KMIS, u : int, y : int) -> list[int]:
  #Conjunto "indesejado" junto a u na partição L
  luy = []  #  L_u(lambda) conjunto resultante
  for v in range(kmis.tamL):
    if v!=u:
      if ((kmis.L[u] & kmis.L[v])).bit_count() < y:
        luy.append(v)
  return luy

def Rv(kmis : KMIS, v : int, k : int) -> list[int]:
  #Conjunto "indesejado" junto a v na partição R
  rvk = []  #R_v(k)
  for u in range(kmis.tamR):
    if u!=v:
      if ((kmis.R[v] & kmis.R[u])).bit_count() < k:
        rvk.append(u)
  return rvk

def Lu_tam(kmis : KMIS, u : int, y : int) -> int:
  luy_tam = 0  #  |L_u(lambda)| valor
  for v in range(kmis.tamL):
    if v!=u:
      if ((kmis.L[u] & kmis.L[v])).bit_count() < y:
        luy_tam += 1
  return luy_tam

def Rv_tam(kmis : KMIS, v : int, k : int) -> int:
  rvk_tam = 0  # |R_v(k)|
  for u in range(kmis.tamR):
    if u!=v:
      if ((kmis.R[v] & kmis.R[u])).bit_count() < k:
        rvk_tam += 1
  return rvk_tam


"""Redução Bogue (2014) |<BR> [Bogue, 2014](#scrollTo=dyXmWyS0zIQS)"""
#===================================================================================
def reducao_Bogue14(kmis_entrada : KMIS) -> KMIS:
  # Esta meio lento, mas já não sei o que melhorar
  kmis = dc(kmis_entrada)
  try:
    lower = kmis_entrada.intersect(kInterEstendida(kmis_entrada))
  except:
    print("Erro na reducao")
    sol = SOLUCAO(kmis_entrada.k, kmis_entrada.tamL)
    for i in range(kmis_entrada.k):
      sol.append(i)
    lower = kmis_entrada.intersect(sol)

  k = kmis_entrada.k
  has_change = True
  while has_change:
    has_change = False
    # Primeira regra
    limite_Lu = kmis.tamL - k
    for u in range(kmis.tamL):
      if((kmis.L[u].bit_count() < lower) or (Lu_tam(kmis, u, lower) > limite_Lu)):
        kmis.remover('L', u)
        has_change = True
        break
    # Segunda regra
    limite_Rv = kmis.tamR - lower
    for v in range(kmis.tamR):
      if((kmis.R[v].bit_count() < k)   or    (Rv_tam(kmis, v, k)  > limite_Rv)):
        kmis.remover('R', v)
        has_change = True
        break

  return kmis


"""Funções de estatísticas dos testes""" # Planos para reduzir isso, dá pra juntar algumas creio
#===============================================================================================
def junta_repeticoes(g):
  return pd.Series({'vmin': g['val'].min(), 'vavg': g['val'].mean(),
                    'vmax': g['val'].max(), 'tavg': g['time'].mean()})
def medias(g):
  return g.mean().rename(lambda x: 'm'+x)  # gera: mvmin, mvmax, mvavg, mtavg
def melhor_por_instancia(g):
  return pd.Series({'vmin_max': g['vmin'].max(),
                    'vavg_max': g['vavg'].max(),
                    'vmax_max': g['vmax'].max(),
                    'tavg_min': g['tavg'].min()})
def limites_argumento(g):
  return pd.Series({'mvmin_min'   : g['mvmin'].min()   , 'mvmin_max'   : g['mvmin'].max(),
                    'mvavg_min'   : g['mvavg'].min()   , 'mvavg_max'   : g['mvavg'].max(),
                    'mvmax_min'   : g['mvmax'].min()   , 'mvmax_max'   : g['mvmax'].max(),
                    'mtavg_min'   : g['mtavg'].min()   , 'mtavg_max'   : g['mtavg'].max(),
                    'cnt_vmin_min': g['cnt_vmin'].min(), 'cnt_vmin_max': g['cnt_vmin'].max(),
                    'cnt_vavg_min': g['cnt_vavg'].min(), 'cnt_vavg_max': g['cnt_vavg'].max(),
                    'cnt_vmax_min': g['cnt_vmax'].min(), 'cnt_vmax_max': g['cnt_vmax'].max(),
                    'cnt_tavg_min': g['cnt_tavg'].min(), 'cnt_tavg_max': g['cnt_tavg'].max(),
                    })

def score_time_on(r):
  return pd.Series(
    {  #mv = media valor  e cnt = count
      'score':
      int(100*(
        10*(((r['mvmin']-r['mvmin_min'])/(r['mvmin_max']-r['mvmin_min'])) if (r['mvmin_max']-r['mvmin_min']) > 0 else 1) +
        20*(((r['mvavg']-r['mvavg_min'])/(r['mvavg_max']-r['mvavg_min'])) if (r['mvavg_max']-r['mvavg_min']) > 0 else 1) +
        10*(((r['mvmax']-r['mvmax_min'])/(r['mvmax_max']-r['mvmax_min'])) if (r['mvmax_max']-r['mvmax_min']) > 0 else 1) +
        10*(((r['mtavg_max']-r['mtavg'])/(r['mtavg_max']-r['mtavg_min'])) if (r['mtavg_max']-r['mtavg_min']) > 0 else 1) +
        10*(((r['cnt_vmin']-r['cnt_vmin_min'])/(r['cnt_vmin_max']-r['cnt_vmin_min'])) if (r['cnt_vmin_max']-r['cnt_vmin_min']) > 0 else 1) +
        20*(((r['cnt_vavg']-r['cnt_vavg_min'])/(r['cnt_vavg_max']-r['cnt_vavg_min'])) if (r['cnt_vavg_max']-r['cnt_vavg_min']) > 0 else 1) +
        10*(((r['cnt_vmax']-r['cnt_vmax_min'])/(r['cnt_vmax_max']-r['cnt_vmax_min'])) if (r['cnt_vmax_max']-r['cnt_vmax_min']) > 0 else 1) +
        10*(((r['cnt_tavg']-r['cnt_tavg_min'])/(r['cnt_tavg_max']-r['cnt_tavg_min'])) if (r['cnt_tavg_max']-r['cnt_tavg_min']) > 0 else 1)
        ))/100
    }
  )

def score_time_off(r):
  return pd.Series(
    {  #mv = media valor  e cnt = count
      'score':
      int(10000*(
        15*(((r['mvmin']-r['mvmin_min'])/(r['mvmin_max']-r['mvmin_min'])) if (r['mvmin_max']-r['mvmin_min']) > 0 else 1) +
        20*(((r['mvavg']-r['mvavg_min'])/(r['mvavg_max']-r['mvavg_min'])) if (r['mvavg_max']-r['mvavg_min']) > 0 else 1) +
        15*(((r['mvmax']-r['mvmax_min'])/(r['mvmax_max']-r['mvmax_min'])) if (r['mvmax_max']-r['mvmax_min']) > 0 else 1) +
        #0*(((r['mtavg_max']-r['mtavg'])/(r['mtavg_max']-r['mtavg_min'])) if (r['mtavg_max']-r['mtavg_min']) > 0 else 1) +
        15*(((r['cnt_vmin']-r['cnt_vmin_min'])/(r['cnt_vmin_max']-r['cnt_vmin_min'])) if (r['cnt_vmin_max']-r['cnt_vmin_min']) > 0 else 1) +
        20*(((r['cnt_vavg']-r['cnt_vavg_min'])/(r['cnt_vavg_max']-r['cnt_vavg_min'])) if (r['cnt_vavg_max']-r['cnt_vavg_min']) > 0 else 1) +
        15*(((r['cnt_vmax']-r['cnt_vmax_min'])/(r['cnt_vmax_max']-r['cnt_vmax_min'])) if (r['cnt_vmax_max']-r['cnt_vmax_min']) > 0 else 1)
        #0*(((r['cnt_tavg']-r['cnt_tavg_min'])/(r['cnt_tavg_max']-r['cnt_tavg_min'])) if (r['cnt_tavg_max']-r['cnt_tavg_min']) > 0 else 1)
        ))/10000
    }
  )
