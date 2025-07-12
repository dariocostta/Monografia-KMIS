from bibkmis.typeskmis import *
from copy import deepcopy as dc
import time
import numpy as np
from numpy.random import choice

"""## **Heurísticas - Base**"""

"""**HG** |<BR> [Bogue, 2013](#scrollTo=ogM8m0ZZUWYq)"""
#======================================================================================
def HG_Bogue13(kmis: KMIS, t_lim: float = 120.0) -> SOLUCAO: #Heuristica Gulosa de Bogue 2013
  L : SOLUCAO = SOLUCAO(kmis.tamL, kmis.k)
  BIN = kmis.Rcompleto #Conjunto R completo
  while(len(L) < kmis.k):
    gJmax, Jmax = -1, -1
    for Sj in range(kmis.tamL):
      if(not L.inL[Sj]):
        gBIN2 = (BIN & kmis.L[Sj]).bit_count() #intersecao & e depois conta num. de elementos
        if(gJmax < gBIN2):
          gJmax, Jmax = gBIN2, Sj

    BIN = BIN & kmis.L[Jmax]
    L.append(Jmax)
  return L

"""**Heuristica Gulosa parcial**
É a "MaxAways" (ou "kInter"), mas aceitando uma solução parcial de partida.
"""

def HG_Bogue13_parcial(kmis: KMIS, L_entrada : SOLUCAO, t_lim: float = 120.0) -> SOLUCAO:
  L = dc(L_entrada)
  BIN1 = kmis.Rcompleto #Conjunto R completo
  for Si in L:
    BIN1 = BIN1 & kmis.L[Si]
  while(len(L) < kmis.k):
    gJmax, Jmax = -1, -1
    for Sj in range(kmis.tamL):
      if(not L.inL[Sj]):
        gBIN2 = (BIN1 & kmis.L[Sj]).bit_count()
        if(gJmax < gBIN2):
          gJmax, Jmax = gBIN2, Sj
    BIN1 = BIN1 & kmis.L[Jmax]
    L.append(Jmax)
  return L

"""**Heuristica Gulosa Estendida**
É a "MaxAways" (ou "kInter"), mas com todos os Si in L testados como primeira escolha.
Utilizada em [José Robertty, 2020](#scrollTo=ausmti4Msbm9) para a Redução das instâncias.
"""
def kInterEstendida(kmis : KMIS, t_lim: float = 120.0) -> SOLUCAO:
  t_inicio = time.time()

  Lb : SOLUCAO = SOLUCAO(kmis.tamL, kmis.k)
  gLb: int     = -1
  L  : SOLUCAO = SOLUCAO(kmis.tamL, kmis.k)
  gL : int     = -1

  for Si in range(kmis.tamL):
    L.reset(Si)
    BIN1 : int = kmis.L[Si]

    while(len(L) < kmis.k):
      gJmax, Jmax = -1, -1
      for Sj in range(kmis.tamL):
        if(not L.inL[Sj]):
          gBIN2 = (BIN1 & kmis.L[Sj]).bit_count()
          if(gJmax < gBIN2):
            gJmax, Jmax = gBIN2, Sj

      BIN1 = BIN1 & kmis.L[Jmax]
      L.append(Jmax)
      gL = BIN1.bit_count()
      if(gL <= gLb): break

    if(gLb < gL):
      Lb, gLb = dc(L), gL

    if(t_lim < (time.time()-t_inicio)): break

  return Lb

"""**LS** |<BR> [Casado, 2022](#scrollTo=vUV56ei_QkEj)"""
#===========================================================================
# Usado como Intensificação
def LocalSearch(kmis: KMIS, L_entrada: SOLUCAO, t_lim: float = 120.0) -> SOLUCAO:
  t_inicio = time.time()
  if(len(L_entrada) < kmis.k): # <---- Não aceita solucao parcial
    L = kInterEstendida(kmis)
  else:
    L = dc(L_entrada)

  gL = kmis.intersect(L) # valor da intersecção da solucao atual

  improve = True
  while(improve):
    improve = False
    EL_sequence = np.random.permutation(kmis.k)# <---- ordem por index da Solucao
    for indexSi in EL_sequence: #Percorrer em ordem randomica
      Si = L[indexSi]
      BIN1 = kmis.Rcompleto #intersecao sem o Si
      for Sk in L:
        if(Sk != Si):
          BIN1 = BIN1 & kmis.L[Sk]

      CL_sequence = np.random.permutation(kmis.tamL) # <---- sorteio por label de L
      for Sj in CL_sequence:  #Percorrer em ordem randomica
        if(not L.inL[Sj]):
          #valor da intersecção trocando i por j
          gBIN2 = (BIN1 & kmis.L[Sj]).bit_count()
          if(gL < gBIN2):  # -> Improve
            L[indexSi] = int(Sj) # SWAP (operador sobrecarregado)
            gL = gBIN2
            improve = True
            break
      if(improve): break

    if(t_lim < (time.time()-t_inicio)): break

  return L

"""**VND** |<BR> [Dias, 2020](#scrollTo=Dub1Q9UAepci)"""
#=======================================================================================
# Usado como Intensificação
def VND(kmis: KMIS, L_entrada: SOLUCAO, t_lim: float = 120.0, rep: int = 3) -> SOLUCAO:
  t_inicio = time.time()
  if(len(L_entrada) < kmis.k):  #<---- Não aceita solucao parcial
    L = kInterEstendida(kmis)
  else:
    L = dc(L_entrada)
  gL: int = kmis.intersect(L)

  f : int = kmis.k - 1 if(kmis.k <= 3) else max(int(0.3*kmis.k), 3)
  idxEL : list[int] = [-1 for _ in range(f)] # index dos escolhidos para sair na iteração atual
  CL    : list[int] = [-1 for _ in range(f)] # Label dos subconjuntos que podem entrar na iteração atual
  EH    : list[bool]= [False for _ in range(kmis.tamL)] # Historico de saida geral "Exit History"

  while(rep > 0):
    choices_made = 0
    for i in range(f):
      # procuro de forma rotativa a partir do index sorteado o primeiro valido da EL
      idxSi = np.random.randint(kmis.k) #Si saida
      safe_count = 0
      while(EH[L[idxSi]] and safe_count < kmis.k):
        idxSi = (idxSi + 1) % kmis.k
        safe_count+=1
      if(safe_count == kmis.k and choices_made == 0):
        return L  # TODOS da solucao foram tentados na remoção.
      else:
        choices_made +=1
      EH[L[idxSi]] = True
      idxEL[i] = idxSi

    for i in range(choices_made):
      L.inL[L[idxEL[i]]] = False  # OPERAÇÃO na ->inL diretamente, "removendo" apenas nela

    BIN1 = kmis.Rcompleto # BIN solucao parcial
    for Sk in L:
      if(L.inL[Sk]):
        BIN1 = BIN1 & kmis.L[Sk]

    for i in range(choices_made):
      Jmax, gJmax = -1, -1
      for Sj in range(kmis.tamL):
        if(not L.inL[Sj]):
          gBIN2 = (BIN1 & kmis.L[Sj]).bit_count()
          if(gJmax < gBIN2):
            Jmax, gJmax = Sj, gBIN2

      CL[i] = Jmax
      L.inL[Jmax] = True # OPERAÇÃO na ->inL diretamente, "adicionando" apenas nela
      BIN1 = BIN1 & kmis.L[Jmax]
    gBIN1 = BIN1.bit_count()

    if(gL < gBIN1):
      gL = gBIN1
      for i in range(choices_made):       # Efetivando a troca
        L.L_linha[idxEL[i]] = CL[i]  # OPERAÇÃO na ->L_linha diretamente
        # A inL já foi alterada na execução das escolhas
    else:
      # Restaurando a inL já que não melhorou
      for i in range(choices_made):
        L.inL[CL[i]]       = False
      for i in range(choices_made):
        L.inL[L[idxEL[i]]] = True

    rep = rep - 1
    if(t_lim < (time.time() - t_inicio)): break
  return L

"""**VND |<BR> Mod. Autor"""
#========================================================================
# Usado como Intensificação
def VND2(kmis: KMIS, L_entrada: SOLUCAO, t_lim: float = 120.0) -> SOLUCAO:
  t_inicio = time.time()
  if(len(L_entrada) < kmis.k):  #<---- Não aceita solucao parcial
    L = kInterEstendida(kmis)
  else:
    L = dc(L_entrada)
  gL: int = kmis.intersect(L)

  f : int = kmis.k - 1 if(kmis.k <= 3) else max(int(0.3*kmis.k), 3)
  idxEL : list[int] = [-1 for _ in range(f)] # index dos escolhidos para sair na iteração atual
  CL    : list[int] = [-1 for _ in range(f)] # Label dos subconjuntos que podem entrar na iteração atual
  insistencia = 5
  while(insistencia>0):
    EH : list[bool]= [False for _ in range(kmis.tamL)] # Historico de saida geral "Exit History"
    rep: int = 3
    while(rep > 0):
      choices_made = 0
      for i in range(f):
        # procuro de forma rotativa a partir do index sorteado o primeiro valido da EL
        idxSi = np.random.randint(kmis.k) #Si saida
        safe_count = 0
        while(EH[L[idxSi]] and safe_count < kmis.k):
          idxSi = (idxSi + 1) % kmis.k
          safe_count+=1
        if(safe_count == kmis.k and choices_made == 0):
          return L  # TODOS da solucao foram tentados na remoção.
        else:
          choices_made +=1
        EH[L[idxSi]] = True
        idxEL[i] = idxSi

      for i in range(choices_made):
        L.inL[L[idxEL[i]]] = False  # OPERAÇÃO na ->inL diretamente, "removendo" apenas nela

      BIN1 = kmis.Rcompleto # BIN solucao parcial
      for Sk in L:
        if(L.inL[Sk]):
          BIN1 = BIN1 & kmis.L[Sk]

      for i in range(choices_made):
        Jmax, gJmax = -1, -1
        for Sj in range(kmis.tamL):
          if(not L.inL[Sj]):
            gBIN2 = (BIN1 & kmis.L[Sj]).bit_count()
            if(gJmax < gBIN2):
              Jmax, gJmax = Sj, gBIN2

        CL[i] = Jmax
        L.inL[Jmax] = True # OPERAÇÃO na ->inL diretamente, "adicionando" apenas nela
        BIN1 = BIN1 & kmis.L[Jmax]
      gBIN1 = BIN1.bit_count()

      if(gL < gBIN1):
        gL = gBIN1
        for i in range(choices_made):       # Efetivando a troca
          L.L_linha[idxEL[i]] = CL[i]  # OPERAÇÃO na ->L_linha diretamente
          # A inL já foi alterada na execução das escolhas
        insistencia = 5
        break
      else:
        # Restaurando a inL já que não melhorou
        for i in range(choices_made):
          L.inL[CL[i]]       = False
        for i in range(choices_made):
          L.inL[L[idxEL[i]]] = True

      rep = rep - 1
      if(t_lim < (time.time() - t_inicio)): break
    insistencia-=1

  return L

"""**Variable Neighborhood Descent 2018** |<BR> [Robertty, 2018](#scrollTo=kpxrB6wQ3C7R)"""
#=======================================================================================
def vizinhancaVND_2018(kmis:KMIS, L_entrada : SOLUCAO, t : int) -> SOLUCAO:
  L = dc(L_entrada)
  gL = kmis.intersect(L)

  EL = dc(L.inL)    #Exit List sem repeticao. Saiu, caso volte, não sai de novo
  while(t > 0):
    indexSi = np.random.randint(kmis.k) #Si saida
    # procuro de forma rotativa a partir do index sorteado o primeiro valido da EL
    while(not EL[L[indexSi]]):
      indexSi = (indexSi + 1) % kmis.k

    Si = L[indexSi]
    EL[Si] = False
    BIN1 = kmis.Rcompleto
    for Sk in L:
      if(Sk != Si):
        BIN1 = BIN1 & kmis.L[Sk]

    Jmax, gJmax = -1, gL
    for Sj in range(kmis.tamL):
      if(not L.inL[Sj]):
        gBIN2 = (BIN1 & kmis.L[Sj]).bit_count()
        if(gJmax < gBIN2):
          Jmax, gJmax = Sj, gBIN2

    if(Jmax != -1):
      L[indexSi] = Jmax #Swap

    t=t-1

  return L

# Usado como Intensificação
def VND_2018(kmis : KMIS, L_entrada: SOLUCAO, t_lim : float = 120.0,
             profundidade : int = 3) -> SOLUCAO:
  tempo_inicio = time.time()
  if(len(L_entrada) < kmis.k):  #<---- Não aceita solucao parcial
    Lb = kInterEstendida(kmis)
  else:
    Lb = dc(L_entrada)

  gLb = kmis.intersect(Lb)
  L :SOLUCAO = dc(L_entrada)
  t = 1
  if(profundidade > kmis.k): profundidade = kmis.k
  while(t <= profundidade):
    L = vizinhancaVND_2018(kmis, Lb, t) #inplace
    gL = kmis.intersect(L)
    if(gLb < gL):
      Lb, gLb = L, gL
      t=1
    else: t=t+1

    if(t_lim < (time.time() - tempo_inicio)): break

  return Lb

"""**TS** |<BR> [Casado, 2022](#scrollTo=vUV56ei_QkEj)"""
#================================================================
class Tabu: #Short Term Memory and Exit List (S\STM) control
  def __init__(self, L : SOLUCAO):
    self.stm_front = 0
    self.STM = []
    self.EL = dc(L.inL)

  def MarkTabu(self, Si : int, Sj : int, tau_int : int) -> None:
    self.EL[Si] = False
    if(len(self.STM)<tau_int):
      self.STM.append(Sj)
    else:
      self.EL[self.STM[self.stm_front]] = True
      self.STM[self.stm_front] = Sj       # enqueue(Sj)
      self.stm_front = (self.stm_front+1) % tau_int

# Usado como Intensificação
def TabuSearch(kmis : KMIS, L_entrada : SOLUCAO, t_lim : float = 120.0,
               tau : float = 0.5, gama : int = 5) -> SOLUCAO:
  t_inicio = time.time()
  if(len(L_entrada) < kmis.k): # <---- Não aceita solucao parcial!
    L = kInterEstendida(kmis)
  else:
    L = dc(L_entrada)
  Lb = dc(L)
  gLb = kmis.intersect(L) # valor da intersecção da melhor solucao
  T = Tabu(L)

  if(kmis.tamL == len(L)): # |L| == k (não existe elemento fora da solução)
    return L
  #                   [1, tau*k, k-1]
  tau_int : int = max(1, min(int(kmis.k*tau + 0.5), kmis.k-1))

  GAMA = 0
  while(GAMA < gama):
    improve = False
    indexSbi, Sbj, gb = -1, -1, -1 # Melhor movimento da iteração, b = best
    E_sequence = np.random.permutation(kmis.k)
    for indexSi in E_sequence:
      if(T.EL[L[indexSi]]):
        Si = L[indexSi]
        BIN1 : int = kmis.Rcompleto
        for Sk in L:
          if(Sk != Si):
            BIN1 = BIN1 & kmis.L[Sk]

        C_sequence = np.random.permutation(kmis.tamL)
        for Sj in C_sequence:
          if(not L.inL[Sj]):
            #valor da intersecção trocando i por j
            gBIN2 = (BIN1 & kmis.L[Sj]).bit_count()
            if(gLb < gBIN2):   # improve
              L[indexSi] = int(Sj) #Swap  (operador sobrecarregado)
              gLb, Lb = gBIN2, dc(L)
              T.MarkTabu(Si, Sj, tau_int)
              improve = True
              GAMA = 0
              break
            elif(gb < gBIN2): # Update Less deteriorating move
              indexSbi, Sbj, gb = indexSi, Sj, gBIN2

      if(improve): break

    if(not improve): # Perform less deteriorating move
      T.MarkTabu(L[indexSbi], Sbj, tau_int)
      L[indexSbi] = int(Sbj)
      GAMA+=1

    if(t_lim < (time.time()-t_inicio)): break

  return Lb

"""**GRASP** |<BR> [Casado, 2022](#scrollTo=vUV56ei_QkEj)"""
#============================ GRASP ==============================================
def GRASP(CONSTRUCAO, INTENSIFICACAO, arg_contrucao, arg_intensificacao,
          kmis : KMIS, maxIter : int = 50, t_lim : float = 120.0) -> SOLUCAO:
  t_inicio = time.time()
  Lb  : SOLUCAO = SOLUCAO(kmis.tamL, kmis.k) # Melhor solucao
  L   : SOLUCAO = SOLUCAO(kmis.tamL, kmis.k)
  gLb : int     = -1
  for ite in range(maxIter):
    CONSTRUCAO(kmis, L, **arg_contrucao) #inplace
    L_2 = INTENSIFICACAO(kmis, L, t_lim-(time.time()-t_inicio), **arg_intensificacao)# new SOLUCAO
    gL_2 = kmis.intersect(L_2)
    if(gLb < gL_2):
      Lb, gLb = L_2, gL_2

    if(t_lim < (time.time()-t_inicio)): break

  return Lb

# ========= Fase de Construcao ==========
# Greedy Random
def CGR(kmis, L, alpha) -> SOLUCAO:
  # g = "função objetivo"
  L.reset(choice(kmis.tamL))
  BIN1 = kmis.L[L[0]]
  while(len(L) < kmis.k):
    # Encontrando g(c) min e max
    gmin, gmax = kmis.tamR, 0
    for c in range(kmis.tamL):
      if(not L.inL[c]):
        g_c = (BIN1 & kmis.L[c]).bit_count()
        if g_c < gmin: gmin = g_c
        if g_c > gmax: gmax = g_c
    # Limite de corte para entrar na RCL
    mu = gmax - alpha*(gmax-gmin)
    # Buscando c com g(c) >= mu, começando de um ponto aleatorio de CL
    start = np.random.randint(kmis.tamL)
    Si : int = -1
    for c in range(kmis.tamL):
      Si = (start+c) % len(kmis.tamL) #index rotativo
      if(not L.inL[Si]):
        if((BIN1 & kmis.L[Si]).bit_count() >= mu):
          L.append(Si) # Primeiro g(Si) >= mu entra
          break

    BIN1 = BIN1 & kmis.L[Si]

  return L

# Random Greedy
def CRG(kmis, L, alpha) -> SOLUCAO:
  # g = "função objetivo"
  L.reset(choice(kmis.tamL))
  BIN1 = kmis.L[L[0]]
  visitado = [False for _ in range(kmis.tamL)] # No lugar de dar dc(inL) k vezes, decidi criar 1 vetor só e resetar ele
  while(len(L) < kmis.k):
    # Numero de escolhas randomicas (alpha*|CL|) da CL
    tamRCL = max(1,  int(alpha*(kmis.tamL - len(L))+0.5))
    # Pegando o argmax g(c), c em RCL
    cmax, gcmax = -1, -1
    while(tamRCL > 0):
      c = np.random.randint(kmis.tamL)
      while(visitado[c] or L.inL[c]): #<- procurando o valido mais proximo do sorteado
        c = (c + 1) % kmis.tamL # rotativo
      gBIN2 = (BIN1 & kmis.L[c]).bit_count()
      if(gcmax < gBIN2):
        gcmax, cmax = gBIN2, c

      visitado[c] = True  #<- Já foi testado
      tamRCL = tamRCL - 1

    L.append(cmax)
    BIN1 = BIN1 & kmis.L[cmax]

    for bit in range(len(visitado)):
      visitado[bit] = False

  return L

#==== Definições do GRASP com variações =============
def GRASP_GR_LS(kmis : KMIS, alpha : float = 0.4, maxIter : int = 50, t_lim : float = 120.0) -> SOLUCAO:
  return GRASP(CGR, LocalSearch, {'alpha':alpha}, {}, kmis, maxIter, t_lim)

def GRASP_RG_LS(kmis : KMIS, alpha : float = 0.4, maxIter : int = 50, t_lim : float = 120.0) -> SOLUCAO:
  return GRASP(CRG, LocalSearch,  {'alpha':alpha}, {}, kmis, maxIter, t_lim)

"""**ANT** |<BR> [Dias, 2022](#scrollTo=WhinMVuS_iPw)"""
#===========================================================================================
def ANT(INTENSIFICACAO, arg_intensificacao,
        kmis : KMIS, alpha: float = 1.0, beta : float = 0.8, rho : float = 0.3,
        q_zero : float = 0.8, qtd_formigas : int = 5, Q_reativo : int = 5,
        maxIter : int = 30, t_lim : float = 120.0) -> SOLUCAO:
  t_inicio = time.time()
  # alpha e beta sao expoente para tau e eta (feromonio e fator guloso intrinseco)
  # rho é o parametro de evaporação
  Lb   : SOLUCAO   = SOLUCAO(kmis.tamL, kmis.k) # Melhor solucao geral
  Ab   : SOLUCAO   = SOLUCAO(kmis.tamL, kmis.k) # Armazena a melhor solução da iteração
  gLb  : int = -1          # valor de L
  # Feromonios deviam ser um parametro, mas por enquanto tá como variavel local
  gHG = kmis.intersect(HG_Bogue13(kmis))
  numArestas = ((kmis.tamL-1)*kmis.tamL)//2 # i < j
  statAresta : list[dict] = [{'tau':1/gHG if gHG > 0 else 0.2, 'SUM':0, 'count':0} for _ in range(numArestas)]
  idxE = lambda i, j:  ((i*(kmis.tamL-1) + j-1 - ((i*(i+1))//2))
            if i<j else (j*(kmis.tamL-1) + i-1 - ((j*(j+1))//2)))

  eta = lambda j, B: (1/((B.bit_count() - (B & kmis.L[j]).bit_count()) + 1))
  # [soma das solucoes, nº de vezes que a formiga i ficou ativa, ultimo valor de solucao]
  statAnt : list[dict[str, int]] = [{'SUM':0, 'count':0, 'lastVal':0} for _ in range(kmis.tamL)]
  A : list[SOLUCAO]  = [SOLUCAO(kmis.tamL, kmis.k) for _ in range(kmis.tamL)] #Ants
  for ite in range(maxIter):
    # Constructive Ant
    # Formigas
    if(ite < Q_reativo):
      activeAnts = range(kmis.tamL)
    elif(ite % 2 == 0):
      activeAnts = choice(kmis.tamL, qtd_formigas, replace = False)
    else:
      mA = [[(statAnt[i]['SUM']/statAnt[i]['count']), i] for i in range(len(statAnt))]
      mA.sort()  # <--------- #TO DO: OTIMIZAR?
      activeAnts = [mA[kmis.tamL - i - 1][1] for i in range(qtd_formigas)]

    gAb = -1
    for ant in activeAnts:
      A[ant].reset(int(ant))

    for ant in activeAnts:
      if(t_lim < (time.time()-t_inicio)):
        if(gAb > -1): break
        if(ite >  0): return Lb

      BINAt = kmis.L[A[ant][0]]
      while(len(A[ant]) < kmis.k):
        if(t_lim < (time.time()-t_inicio)):
          if(gAb > -1): break
          if(ite >  0): return Lb

        # Regra da Proporcionalidade Pseudoaleatória (ACS)
        r = np.random.random()
        Jnew = -1
        if(r <= q_zero):
          # argmax
          argmax, valmax = -1, -1
          for k in range(kmis.tamL):
            if(not A[ant].inL[k]):
              val = (statAresta[idxE(A[ant][len(A[ant])-1], k)]['tau']) * ((eta(k, BINAt))**beta)
              if valmax < val:
                argmax, valmax =  k, val
          Jnew = argmax
        else:
          # Regra da Proporcionalidade Aleátoria (AS)
          sumTauEta = 0
          for j in range(kmis.tamL):
            if(not A[ant].inL[j]):
              sumTauEta+=((statAresta[idxE(A[ant][len(A[ant])-1], j)]['tau'])**alpha) * ((eta(j, BINAt))**beta)

          ptij = np.random.random()*sumTauEta # Sorteio no intervalo [0, sumTauEta)
          Jas = -1
          sumTauEta = 0
          for j in range(kmis.tamL):
            if(not A[ant].inL[j]):
              sumTauEta+=((statAresta[idxE(A[ant][len(A[ant])-1], j)]['tau'])**alpha) * ((eta(j, BINAt))**beta)
              if sumTauEta > ptij:
                Jas = j
                break # Quando a soma passa de ptij encontramos o j escolhido
          Jnew = Jas
        A[ant].append(Jnew)
        BINAt = BINAt & kmis.L[Jnew]
      if(len(A[ant]) == kmis.k):
        statAnt[ant]['lastVal'] = BINAt.bit_count()
        statAnt[ant]['SUM']   += statAnt[ant]['lastVal']  # soma das soluções da formiga t
        statAnt[ant]['count'] += 1   # quantas vezes a formiga t foi ativa
        if(gAb < statAnt[ant]['lastVal']):
          Ab, gAb = A[ant], statAnt[ant]['lastVal']

    # Intensificação ----------- intensificar a melhor da itereção
    L_intensificada = INTENSIFICACAO(kmis, Ab, t_lim-(time.time()-t_inicio), **arg_intensificacao)
    gL_intensificada = kmis.intersect(L_intensificada)
    if(gLb < gL_intensificada):
      Lb, gLb = L_intensificada, gL_intensificada

    if(t_lim < (time.time()-t_inicio)): break

    #-------------- Feromonios    ----------
    for ant in activeAnts:
      if(t_lim < (time.time()-t_inicio)): break
      if(len(A[ant]) == kmis.k):
        # Info para att dos feromonios
        for i in range(len(A[ant])-1): #percorrendo todos os pares (i, j) \in E da solucao At
          for j in range(i+1, len(A[ant])):  #A[ant][i+1:]
            statAresta[idxE(A[ant][i], A[ant][j])]['SUM']+=statAnt[ant]['lastVal']
            statAresta[idxE(A[ant][i], A[ant][j])]['count']+= 1

    for i in range(kmis.tamL-1):
      if(t_lim < (time.time()-t_inicio)): break
      for j in range(i+1, kmis.tamL):
        idx = idxE(i, j)
        deltaij = 0
        if(gLb>0 and statAresta[idx]['count'] > 0):
          deltaij = (statAresta[idx]['SUM']/statAresta[idx]['count'])/gLb
        statAresta[idx]['tau'] = (1 - rho)*(statAresta[idx]['tau']) + rho*(deltaij)

        statAresta[idx]['SUM']   = 0  # Resetando as info do delta, pois é por iteração
        statAresta[idx]['count'] = 0

  return Lb

#========  ANT Definição com LocalSearch  =================
def ANT_LS(kmis : KMIS, alpha: float = 1.0, beta : float = 0.8, rho : float = 0.3,
           q_zero : float = 0.8, qtd_formigas : int = 5, Q_reativo : int = 5,
           maxIter : int = 30, t_lim : float = 120.0) -> SOLUCAO:

  return ANT(LocalSearch, {}, kmis, alpha, beta, rho, q_zero, qtd_formigas, Q_reativo, maxIter, t_lim)

"""**ANT2** |<BR> Mod. Autor"""
#==========================================================================================================
def ANT2(INTENSIFICACAO, arg_intensificacao,
        kmis : KMIS, alpha: float = 1.0, beta : float = 0.8, rho : float = 0.3,
        q_zero : float = 0.8, qtd_formigas_p : float = 0.1, maxIter : int = 30, t_lim : float = 120.0) -> SOLUCAO:
  t_inicio = time.time()
  # alpha e beta sao expoente para tau e eta (feromonio e fator guloso intrinseco)
  # rho é o parametro de evaporação
  Lb   : SOLUCAO   = SOLUCAO(kmis.tamL, kmis.k) # melhor solucao
  Ab   : SOLUCAO   = SOLUCAO(kmis.tamL, kmis.k) # Armazena a melhor solução da iteração
  gLb  : int = -1          # valor de L
  # Feromonios deviam ser um parametro, mas por enquanto tá como variavel local
  gHG = kmis.intersect(HG_Bogue13(kmis))
  numArestas = ((kmis.tamL-1)*kmis.tamL)//2 # i < j
  statAresta : list[dict] = [{'tau':1/gHG if gHG > 0 else 0.2, 'SUM':0, 'count':0} for _ in range(numArestas)]
  idxE = lambda i, j:  ((i*(kmis.tamL-1) + j-1 - ((i*(i+1))//2))
            if i<j else (j*(kmis.tamL-1) + i-1 - ((j*(j+1))//2)))

  eta = lambda j, B: (1/((B.bit_count() - (B & kmis.L[j]).bit_count()) + 1))
  # [soma das solucoes, nº de vezes que a formiga i ficou ativa, ultimo valor de solucao]
  statAnt : list[dict[str, int]] = [{'SUM':0, 'count':0, 'lastVal':0} for _ in range(kmis.tamL)]
  A : list[SOLUCAO]  = [SOLUCAO(kmis.tamL, kmis.k) for _ in range(kmis.tamL)] #Ants
  qtd_formigas : int = min(max(2, int((kmis.tamL * qtd_formigas_p) + 0.5)), kmis.tamL)

  for ite in range(maxIter):
    # Constructive Ant
    # Formigas
    if(ite == 0):
      activeAnts = range(kmis.tamL)
    elif(ite % 2 == 0):
      activeAnts = choice(kmis.tamL, qtd_formigas, replace = False)
    else:
      mA = [[(statAnt[i]['SUM']/statAnt[i]['count']), i] for i in range(len(statAnt))]
      mA.sort()  # <--------- #TO DO: OTIMIZAR?
      activeAnts = [mA[kmis.tamL - i - 1][1] for i in range(qtd_formigas)]

    gAb = -1
    for ant in activeAnts:
      A[ant].reset(int(ant))

    for ant in activeAnts:
      if(t_lim < (time.time()-t_inicio)):
        if(gAb > -1): break
        if(ite >  0): return Lb

      BINAt = kmis.L[A[ant][0]]
      while(len(A[ant]) < kmis.k):
        if(t_lim < (time.time()-t_inicio)):
          if(gAb > -1): break
          if(ite >  0): return Lb

        # Regra da Proporcionalidade Pseudoaleatória (ACS)
        r = np.random.random()
        idx_r = np.random.randint(len(A[ant]))  #idx de L que vai ser usado para o tau! Aleatorio!
        Jnew = -1
        if(r <= q_zero):
          # argmax
          argmax, valmax = -1, -1
          for k in range(kmis.tamL):
            if(not A[ant].inL[k]):
              val = (statAresta[idxE(A[ant][idx_r], k)]['tau']) * ((eta(k, BINAt))**beta)
              if valmax < val:
                argmax, valmax =  k, val
          Jnew = argmax
        else:
          # Regra da Proporcionalidade Aleátoria (AS)
          sumTauEta = 0
          for j in range(kmis.tamL):
            if(not A[ant].inL[j]):
              sumTauEta+=((statAresta[idxE(A[ant][idx_r], j)]['tau'])**alpha) * ((eta(j, BINAt))**beta)

          ptij = np.random.random()*sumTauEta # Sorteio no intervalo [0, sumTauEta)
          Jas = -1
          sumTauEta = 0
          for j in range(kmis.tamL):
            if(not A[ant].inL[j]):
              sumTauEta+=((statAresta[idxE(A[ant][idx_r], j)]['tau'])**alpha) * ((eta(j, BINAt))**beta)
              if sumTauEta > ptij:
                Jas = j
                break # Quando a soma passa de ptij encontramos o j escolhido
          Jnew = Jas
        A[ant].append(Jnew)
        BINAt = BINAt & kmis.L[Jnew]
      if(len(A[ant]) == kmis.k):
        statAnt[ant]['lastVal'] = BINAt.bit_count()
        statAnt[ant]['SUM']   += statAnt[ant]['lastVal']  # soma das soluções da formiga t
        statAnt[ant]['count'] += 1   # quantas vezes a formiga t foi ativa
        if(gAb < statAnt[ant]['lastVal']):
          Ab, gAb = A[ant], statAnt[ant]['lastVal']

    # Intensificação ----------- intensificar a melhor da itereção
    L_intensificada = INTENSIFICACAO(kmis, Ab, t_lim-(time.time()-t_inicio), **arg_intensificacao)
    gL_intensificada = kmis.intersect(L_intensificada)
    if(gLb < gL_intensificada):
      Lb, gLb = L_intensificada, gL_intensificada

    if(t_lim < (time.time()-t_inicio)): break

    # Feromonios    ----------
    for ant in activeAnts:
      if(t_lim < (time.time()-t_inicio)): break
      if(len(A[ant]) == kmis.k):
        # Info para att dos feromonios
        for i in range(len(A[ant])-1): #percorrendo todos os pares (i, j) \in E da solucao At
          for j in range(i+1, len(A[ant])):  #A[ant][i+1:]
            statAresta[idxE(A[ant][i], A[ant][j])]['SUM']+=statAnt[ant]['lastVal']
            statAresta[idxE(A[ant][i], A[ant][j])]['count']+= 1

    for i in range(kmis.tamL-1):
      if(t_lim < (time.time()-t_inicio)): break
      for j in range(i+1, kmis.tamL):
        idx = idxE(i, j)
        deltaij = 0
        if(gLb>0 and statAresta[idx]['count'] > 0):
          deltaij = (statAresta[idx]['SUM']/statAresta[idx]['count'])/gLb
        statAresta[idx]['tau'] = (1 - rho)*(statAresta[idx]['tau']) + rho*(deltaij)

        statAresta[idx]['SUM']   = 0  # Resetando as info do delta, pois é por iteração
        statAresta[idx]['count'] = 0
  return Lb

"""## **Heurísticas - Mista**"""

"""#GRASP+TS |  <BR> [Casado, 2022](#scrollTo=vUV56ei_QkEj)"""
#=============================================================================================
def GRASP_RG_TS(kmis : KMIS, alpha : float = 0.4, tau : float = 0.5,
                gama : int = 10, maxIter : int = 50, t_lim : float = 120.0) -> SOLUCAO:

  return GRASP(CONSTRUCAO=CRG, INTENSIFICACAO=TabuSearch,
               arg_contrucao={'alpha':alpha}, arg_intensificacao={'tau':tau, 'gama':gama},
               kmis=kmis, maxIter=maxIter, t_lim=t_lim)

"""ANT+TS |<BR> Mod. Autor"""
#======================================================================================
def ANT_TS(kmis : KMIS, alpha: float = 1.0, beta : float = 0.8, rho : float = 0.3,
           q_zero : float = 0.8, qtd_formigas : int = 5, Q_reativo : int = 5,
           tau : float = 0.5, gama : int = 5,
           maxIter : int = 30, t_lim : float = 120.0) -> SOLUCAO:

  return ANT(INTENSIFICACAO=TabuSearch, arg_intensificacao={'tau':tau, 'gama':gama},
             kmis=kmis, alpha=alpha, beta=beta, rho=rho, q_zero=q_zero,
             qtd_formigas=qtd_formigas, Q_reativo=Q_reativo, maxIter=maxIter, t_lim=t_lim)

"""ANT2+TS |<BR> Mod. Autor"""
#================================================================================================
def ANT2_TS(kmis : KMIS, alpha: float = 1.0, beta : float = 0.8, rho : float = 0.3,
           q_zero : float = 0.8, qtd_formigas_p :float = 0.1, tau : float = 0.5, gama : int = 5,
           maxIter : int = 30, t_lim : float = 120.0) -> SOLUCAO:

  return ANT2(INTENSIFICACAO=TabuSearch, arg_intensificacao={'tau':tau, 'gama':gama},
              kmis=kmis, alpha=alpha, beta=beta, rho=rho, q_zero=q_zero,
              qtd_formigas_p=qtd_formigas_p, maxIter=maxIter, t_lim=t_lim)

"""GRASP+VND |  <BR> Mod. Autor"""
#============================================================================
def GRASP_RG_VND(kmis : KMIS, alpha : float = 0.4,
                 maxIter : int = 50, t_lim : float = 120.0) -> SOLUCAO:

  return GRASP(CONSTRUCAO=CRG, INTENSIFICACAO=VND, arg_contrucao={'alpha':alpha}, arg_intensificacao={},
               kmis=kmis, maxIter=maxIter, t_lim=t_lim)

"""GRASP+VND2 |  <BR> Mod. Autor"""
#============================================================================
def GRASP_RG_VND2(kmis : KMIS, alpha : float = 0.4,
                 maxIter : int = 50, t_lim : float = 120.0) -> SOLUCAO:

  return GRASP(CONSTRUCAO=CRG, INTENSIFICACAO=VND2, arg_contrucao={'alpha':alpha}, arg_intensificacao={},
               kmis=kmis, maxIter=maxIter, t_lim=t_lim)

"""ANT+VND |<BR> [Dias, 2022](#scrollTo=WhinMVuS_iPw)"""
#=====================================================================================
def ANT_VND(kmis : KMIS, alpha: float = 1.0, beta : float = 0.8, rho : float = 0.3,
            q_zero : float = 0.8, qtd_formigas : int = 5, Q_reativo : int = 5,
            maxIter : int = 30, t_lim : float = 120.0) -> SOLUCAO:

  return ANT(INTENSIFICACAO=VND, arg_intensificacao={}, kmis=kmis, alpha=alpha, beta=beta,
             rho=rho, q_zero=q_zero, qtd_formigas=qtd_formigas,
             Q_reativo=Q_reativo, maxIter=maxIter, t_lim=t_lim)

"""ANT+VND2 |<BR> Mod. Autor"""
#=============================================================================================
def ANT_VND2(kmis : KMIS, alpha: float = 1.0, beta : float = 0.8, rho : float = 0.3,
            q_zero : float = 0.8, qtd_formigas : int = 5, Q_reativo : int = 5,
            maxIter : int = 30, t_lim : float = 120.0) -> SOLUCAO:

  return ANT(INTENSIFICACAO = VND2, arg_intensificacao={}, kmis = kmis,
             alpha=alpha, beta=beta, rho=rho, q_zero=q_zero, qtd_formigas=qtd_formigas,
             Q_reativo=Q_reativo, maxIter=maxIter, t_lim=t_lim)

"""ANT2+VND |<BR> Mod. Autor"""
#==========================================================================================================
def ANT2_VND(kmis : KMIS, alpha: float = 1.0, beta : float = 0.8, rho : float = 0.3,
            q_zero : float = 0.8, qtd_formigas_p : float = 0.1, maxIter : int = 30, t_lim : float = 120.0) -> SOLUCAO:

  return ANT2(INTENSIFICACAO=VND, arg_intensificacao={}, kmis=kmis, alpha=alpha,
              beta=beta, rho=rho, q_zero=q_zero, qtd_formigas_p=qtd_formigas_p,
              maxIter=maxIter, t_lim=t_lim)

"""ANT2+VND2 |<BR> Mod. Autor"""
#================================================================================================================
def ANT2_VND2(kmis : KMIS, alpha: float = 1.0, beta : float = 0.8, rho : float = 0.3,
            q_zero : float = 0.8, qtd_formigas_p : float = 0.1, maxIter : int = 30, t_lim : float = 120.0) -> SOLUCAO:

  return ANT2(INTENSIFICACAO=VND2, arg_intensificacao={}, kmis=kmis, alpha=alpha,
              beta=beta, rho=rho, q_zero=q_zero, qtd_formigas_p=qtd_formigas_p,
              maxIter=maxIter, t_lim=t_lim)

# Dicionario agregador das funções (global)
Heuristicas = {
    'KIEst'       : kInterEstendida,
    'HG'          : HG_Bogue13,
    'VND'         : VND,
    'LS'          : LocalSearch,
    'TS'          : TabuSearch,
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
