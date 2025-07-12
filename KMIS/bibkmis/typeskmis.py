from copy import deepcopy as dc

"""## Definição do problema"""
# @title
# ###Solução com teste de consistencia
# #============================================================================
class SOLUCAO:
  """
  Representa uma solução parcial ou completa para o kMIS.
  Com muito mais eficiência que uma lista simples.
  ### Atributos
  `L_linha (list[int])`: Labels dos escolhidos. Tamanho fixo k.
  `inL (list[bool])`: Indica se o subconjunto i foi incluído.
  `lenL (int)`: Número atual de subconjuntos inseridos.
  ### Métodos
  `__getitem__(index)`: Retorna o subconjunto na posição `index`.
  `__setitem__(index, Sj)`: Swap do `L_linha[index]` por `Sj`.
  `__len__()`: Retorna o tamanho atual da solução (lenL).
  `__iter__()`: Itera pelos subconjuntos atualmente na solução.
  `append(Si)`: Adiciona `Si` à próxima posição disponível.
  `remove(index)`: Remove o subconjunto na posição `index`.
  `reset(init=-1)`: Reseta a solução; se `init` for passado,
              adiciona-o.
  `set_inL()`: Recalcula o vetor `inL` com base em `L_linha`. Evite usar!
  ### Observações
      Não realiza cálculos de BIN ou valor de solução.
  Cada heurística lida com isso individualmente.
      O controle do estado da solução é tarefa por `lenL` e `inL`,
  evitando redimensionamentos de lista.
  """
  def __init__(self, tamL : int, k : int):
    self.L_linha   : list[int]  = [-1 for i in range(k)]
    self.inL : list[bool] = [False for i in range(tamL)]
    self.lenL: int        = 0

  def __setitem__(self, indexSi : int, Sj : int) -> None:
    """L_linha[indexSi] = Sj, faz o swap e atualiza inL"""
    if indexSi >= self.lenL or indexSi<0:
      print('ERRO_SETITEM_index_invalido')
      return
    if Sj >= len(self.inL) or Sj < 0:
      print('ERRO_SETITEM_Sj_invalido')
      return
    Si = self.L_linha[indexSi]
    self.L_linha[indexSi] = Sj
    self.inL[Si]  = False
    self.inL[Sj]  = True

  def append(self, Si : int) -> None:
    if self.lenL == len(self.L_linha):
      print('ERRO_APPEND_ja_tamanho_maximo')
      return
    if Si >= len(self.inL) or Si < 0:
      print('ERRO_APPEND_Si_invalido')
      return
    self.L_linha[self.lenL] = Si
    self.inL[Si] = True
    self.lenL+=1

  def remove(self, index : int) -> None:
    if index >= self.lenL or index<0:
      print('ERRO_REMOVE_index_invalido')
      return
    Si = self.L_linha[index]
    self.L_linha[index] = self.L_linha[self.lenL-1]
    self.inL[Si] = False
    self.lenL = self.lenL - 1

  def reset(self, Si : int = -1) -> None:
    if Si != -1:
      if Si >= len(self.inL) or Si < 0:
        print('ERRO_RESET_Si_invalido')
        return
      self.L_linha[0] = Si
      self.lenL  = 1
      for bit in range(len(self.inL)):
        self.inL[bit] = False
      self.inL[Si] = True
    else:
      self.lenL  = 0
      for bit in range(len(self.inL)):
        self.inL[bit] = False

  def __getitem__(self, index : int) -> int:
    if index >= self.lenL or index<0:
      print('ERRO_GETITEM_index_invalido')
      return -1
    return self.L_linha[index]

  def __len__(self) -> int:
    return self.lenL
  def __iter__(self): #percorrer o vetor L_linha, labels da solucao
    for i in range(self.lenL):  # somente até o tamanho real da solução
      yield self.L_linha[i]
  def __str__(self):
    return f'{self.L_linha[:self.lenL]}'
  def __repr__(self):
    return f'{self.L_linha[:self.lenL]}'

  # def randomSol(self): #Gera uma solucao randomica
  #   self.lenL = len(self.L_linha)
  #   for idx in range(len(self.L_linha)):
  #     #poderia ser choice(tamL, k, replace=False), mas alocaria mais memoria
  #     Si = int(np.random.randint(len(self.inL)))
  #     while(self.inL[Si]):
  #       Si = (Si+1) % (len(self.inL))
  #     self.L_linha[idx]  = Si
  #     self.inL[Si] = True

  # def set_inL(self) -> None:
  #   """Se está usando isso aqui, algo deu errado"""
  #   for i in range(len(self.inL)):
  #     self.inL[i] = False
  #   for i in range(self.lenL):
  #     self.inL[self.L_linha[i]]=True

# @title **Máxima Interscção de k-Subconjuntos**
#=========================================================================
class KMIS:
  """
    Representa uma instância do problema kMIS.
    ### Atributos
    `tamR (int)`: Tamanho do conjunto universo R (|R|).
    `tamL (int)`: Número total de subconjuntos disponíveis (|L|).
    `p (float)`: Densidade de arestas (proporção de
            inclusão de elementos).
    `k (int)`: Quantidade de subconjuntos a serem escolhidos.
    `L (list[int])`: Lista de subconjuntos representados como `int`.
    `R (list[int])`: Representação transposta — elementos
            apontando para subconjuntos.
    `Llabel (list[int])`: Rótulos originais dos subconjuntos
            (usado após redução).
    `Rlabel (list[int])`: Rótulos originais dos elementos.
    `Rcompleto (int)`: Um `int` com todos os bits ativados
            (equiv. a conjunto R completo).
    ### Métodos
    `intersect(solucao, opt=0)`: Calcula interseção da
            solução. Se `opt=0`, retorna |∩Si|; senão, retorna o conjunto.
    `_setR()`: Preenche `R` com base nos subconjuntos `L`.
    `remover(opt='L', u=0)`: Remove subconjunto (opt='L')
            ou elemento (opt='R') de índice `u`.
    `__str__()`: Exibe resumo da instância (|L|, |R|, p, k).
    `__repr__()`: Idêntico ao `__str__()`.
    ### Observações
        - Os subconjuntos são representados como inteiros
        cuja forma binária indica quais elementos do conjunto
        R estão presentes.
        - A operação de interseção usa um operador
        bitwise eficiente.
  """
  def __init__(self, tamL=0, tamR=0, p : float = 0.0, k=0, L=[]):
    self.k    : int = k
    self.L    : list[int] = dc(L) # Criando copia para evitar problema de acesso
    self.tamR : int = tamR

    self.tamL : int = tamL
    self.p    : float = p  #Densidade de arestas [tamL*tamR * p são escolhidas]
    self.R    : list[int] = [0 for _ in range(tamR)]   # se fez necessario para a redução, estava muito lenta
    self.Llabel : list[int] = [i for i in range(tamL)] # Servem de referencia nas instancias reduzidas
    self.Rlabel : list[int] = [i for i in range(tamR)] # para saber quem era cada indice no original
    self.Rcompleto : int = ((1<<tamR)-1)  # equivalente a bitset.set(), todos os bits = 1

    if len(self.L) == 0:
      self.L = [0 for _ in range(tamL)]
    else:
      self._setR()

  def intersect(self, solucao : SOLUCAO, opt : int = 0) -> int:
    """Realiza a intersecao, retorna |∩Si| ou ∩Si.
      #### Argumentos:
      ` solucao (SOLUCAO)`: Auto-explicativo
      ` opt (int)`: Se `opt=0`, retorna |∩Si|; senão, retorna o conjunto """
    if(len(solucao) < 1):
      return 0

    BIN = self.Rcompleto
    for Si in solucao:
      BIN = BIN & self.L[Si] # <-Comparação bit a bit
    if(opt == 0):
      return BIN.bit_count() # Tamanho |inter(Si)|
    else:
      return BIN # Conjunto inter(Si)

  def _setR(self) -> None:
    for j in range(self.tamR):
      self.R[j] = 0
    for i in range(self.tamL): #i in L
      str_BIN_Si = (bin(self.L[i])[2:]).zfill(self.tamR)[::-1] #<- [::-1] inverte a ordem da string, para acessar e_1 -> e_tamR
      for j in range(self.tamR): # j in R
        if(str_BIN_Si[j] == '1'):
          self.R[j] +=(1<<(i))

  def remover(self, opt : str = 'L', u : int = 0) -> None:
    """#### Argumentos:
      `opt (str)`: Indicando de qual partição 'L' ou 'R'
      `u (inteiro)`: Index do elemento a ser removido (não o label)
    """
    if opt == 'L':
      self.L      = self.L[:u]      + self.L[u+1:]
      self.Llabel = self.Llabel[:u] + self.Llabel[u+1:]

      for v in range(self.tamR):
        binRv = (bin(self.R[v])[2:]).zfill(self.tamL)
        self.R[v] = int('0b0' + (binRv[:self.tamL-u-1] + binRv[self.tamL-u:]), 2)

      self.tamL = self.tamL - 1

    else:
      self.R      = self.R[:u]      + self.R[u+1:]
      self.Rlabel = self.Rlabel[:u] + self.Rlabel[u+1:]

      for v in range(self.tamL):
        binSv = (bin(self.L[v])[2:]).zfill(self.tamR)
        self.L[v] = int('0b0' + (binSv[:self.tamR-u-1] + binSv[self.tamR-u:]), 2)
        #0b0 <- esse 0 é para caso tamR passe a ser 0 não dê erro

      self.tamR = self.tamR-1

  def __str__(self):
    return f"|L|={self.tamL} \t|R|={self.tamR} \tp={self.p:.4f} \tk={self.k}"#\nL[:3]={[[si, (bin(si)[2:]).zfill(self.tamR)[::-1]] for si in self.L[:3]]}
  def __repr__(self):
    return f"|L|={self.tamL} \t|R|={self.tamR} \tp={self.p:.4f} \tk={self.k}"#\nL[:3]={[[si, (bin(si)[2:]).zfill(self.tamR)[::-1]] for si in self.L[:3]]}
  
  __all__ = ['SOLUCAO', 'KMIS']