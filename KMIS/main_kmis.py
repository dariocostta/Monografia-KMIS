"""Heuristicas_Aplicadas_kMIS.py

## Aplicação de Heuristicas ao kMIS
**Autor**: Dario Filipe da Silva Costa
**Instituição**: Universidade Federal do Ceará
Trabalho de conclusão de curso, para obtenção de bacharel em Matemática Industrial.
"""

## Bibliotecas
from bibkmis.heuristicaskmis import *
from bibkmis.auxkmis import *
from bibkmis.typeskmis import SOLUCAO, KMIS


import time
import numpy as np              # Principal para ferramentas matematicas
from copy import deepcopy as dc # Usado para passar vetor sem dar problema de acesso
import pandas as pd             # DataFrames
import matplotlib.pyplot as plt
from itertools import product   # Combinação de vetores, para parametros
from tqdm import tqdm           # Barrinha de progresso
import ast                      # Ler os litearias de tipos simples
import os                       # Controle de pastas
import sys                      # Modificar o nome da aba no PowerShell (Local)
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED    # Dividir entre nucleos

path_P = os.path.dirname(os.path.abspath(__file__))+"/arquivos_principais/"

# Pequena alteração no titulo do PowerSheel.
if os.name == 'nt':
  script_name = os.path.basename(sys.argv[0])

  command = (f'powershell -command "function Set-ConsoleTitle '+
             f'{{ $host.ui.rawui.windowtitle = \'{script_name}\' }}" ; Set-ConsoleTitle"')
  os.system(command)

#Código principal (ainda estranho, pois eram varias celulas no notebook)
def main():
  ## **Instâncias**
  #======================================================================================                     <--------- GERAR INSTANCIAS
  # Carrega as instâncias do arquivo 'instancias.csv' ou gera novas seguindo os parametros abaixo.
  # ====== Inicialização dos Dados ==========
  tamL_list = [40, 60, 80, 100, 140, 180, 200, 240, 280, 300]
  instancias_por_classe = 2

  # ======= Geração de Instâncias ==========
  dfI = pd.DataFrame()
  if get_boolean_input('Gerar instâncias novas?', 'Novas I'):
    dictI = {
      "id": [], "kmis": [], "p": [], "k": [],
      "|L|": [], "|R|": [], "L": [], 'temSol': [], 'classe': []
    }
    dfI = pd.DataFrame(columns = list(dictI.keys()))
    for tamL in tamL_list:
      for tamR in [int(tamL * 0.8 + 0.5), tamL, int(tamL * 1.25 + 0.5)]:
        for pClass, kClass in classes:
            for numInst in range(instancias_por_classe):
              p = (pClass + 0.2 * np.random.random())
              k = int((kClass + 0.2 * np.random.random()) * tamL + 0.5)
              kmis = geraKMIS(tamL, tamR, p, k)

              temSol = any(r.bit_count() >= k for r in kmis.R)

              dadosI = {
                "id": f"{classes[(pClass, kClass)]}p{int(p*100)}k{k}L{tamL}R{tamR}_{numInst}",
                "kmis": kmis, "p": p, "k": k, "|L|": tamL,
                "|R|": tamR, "L": kmis.L, 'temSol': temSol,
                'classe': classes[(pClass, kClass)]
              }
              dictAppend(dictI, dadosI)

    dfI = pd.DataFrame(dictI)
  else:
    dictI = {
      "id": [], "kmis": [], "p": [], "k": [],
      "|L|": [], "|R|": [], "L": [], 'temSol': [], 'classe': []
    }
    dfI = pd.DataFrame(columns = list(dictI.keys()))
    # ==== Carregamento de Instâncias Salvas =====
    conv = {
      'L'     : ast.literal_eval,
      'temSol': ast.literal_eval,
      'L_b14' : ast.literal_eval,
      'Llabel': ast.literal_eval,
      'Rlabel': ast.literal_eval
    }
    try:
      dfI = pd.read_csv(path_P+'instancias.csv', converters=conv)
      print(f'Leitura de instancias.csv ({dfI.shape[0]} linhas) bem sucedida.')
    except:
      print('\n\n\t\tArquivo instancias não encontrado!!\n\n')
      assert dfI.shape[0]>0 , "Sem instâncias não continua! Peque o arquivo 'instancias.csv'."

    # Reinstanciar objetos KMIS a partir das linhas do CSV
    dictI['kmis_b14'] = []
    for _, row in dfI.iterrows():
      kmis = KMIS(row['|L|'], row['|R|'], row['p'], row['k'], row['L'])
      kmis_reduzido = KMIS(row['|L|_b14'], row['|R|_b14'], row['p'], row['k'], row['L_b14'])
      kmis_reduzido.Llabel = row['Llabel_b14']
      kmis_reduzido.Rlabel = row['Rlabel_b14']
      dictI['kmis'].append(kmis)
      dictI['kmis_b14'].append(kmis_reduzido)

    dfI['kmis'] = dictI['kmis']
    dfI['kmis_b14'] = dictI['kmis_b14']
    # dfI.drop(['L', 'L_b14', 'Llabel_b14', 'Rlabel_b14'], axis = 1, inplace=True)

  assert isinstance(dfI, pd.DataFrame), '\t ⚠️ DataFrame de Instâncias não definido.'
  tamanhos_L = dfI[dfI['temSol']]['|L|'].value_counts().reset_index().sort_values(by='|L|')
  MAX_TAMANHO_L : int = int(tamanhos_L['|L|'].max())


  """**Executar Redução**"""
  #===============================================================================                    <------- Executar Redução
  dictIR = {'kmis_b14':[], '|L|_b14':[],'|R|_b14':[], 'L_b14':[], 'tempo_reducao':[],
            'Llabel_b14':[], 'Rlabel_b14':[], 'classe_b14':[], 'p_b14':[]}
  if get_boolean_input('Reduzir as instâncias?', 'Redução'):
    total_iters = dfI[dfI['temSol']][:].shape[0]  # Total barra de progresso
    with tqdm(total=total_iters, smoothing=0.05, desc="Reduzindo instâncias") as pbar:
      for id, i in dfI.iterrows():
        if i['temSol']:
          t_0 = time.time()
          kmis_new = reducao_Bogue14(i.kmis)
          t_f = time.time()-t_0
          pbar.update(1)
          class_new, p_new = get_Classe_e_p(kmis_new)
          kmis_new.p = p_new
        else:
          kmis_new = dc(i.kmis)
          t_f, class_new, p_new = -1, '-1', -1.0

        dadosIb14 = {'kmis_b14':kmis_new, '|L|_b14':kmis_new.tamL,
                    '|R|_b14':kmis_new.tamR, 'L_b14':kmis_new.L, 'tempo_reducao':t_f,
                    'Llabel_b14':kmis_new.Llabel, 'Rlabel_b14':kmis_new.Rlabel,
                    'classe_b14':class_new, 'p_b14':p_new}
        dictAppend(dictIR, dadosIb14)

    for i in dictIR:
        dfI[i] = dictIR[i]
    colunas = ['id', 'p', 'k', '|L|', '|R|', 'L', 'temSol', 'classe', '|L|_b14', '|R|_b14',
              'L_b14', 'tempo_reducao', 'Llabel_b14', 'Rlabel_b14', 'classe_b14', 'p_b14']
    dfI[colunas].to_csv(path_P+'instancias.csv', index=False)
    print('Redução salva com sucesso!')
    # dfI.drop(['L', 'L_b14', 'Llabel_b14', 'Rlabel_b14'], axis = 1, inplace=True)



  # @title --------- TESTE ZONE -----------
  #=================================================================================================
  if False: #get_boolean_input('Ativar Teste Zone?', 'Teste Zone'):
    num_instancias = 5
    reps = 3
    OLD = False
    tamL, tamR = 100, 120
    if not OLD:
      kmis_teste : list[str | KMIS] = [f'({tamL}, {tamR})_id_{int(np.random.random()*1000000)}']
      for _ in range(num_instancias):
        # p, k = 0.6+0.3*np.random.random(), int(tamL*0.3+(tamL*0.3)*np.random.random()+0.5)
        # p, k = 0.6, 30
        # kmis_teste.append(geraKMIS(tamL, tamR, p, k))
        # kmis_teste.append(dfI[(dfI['|L|']==300) & (dfI['temSol'])].iloc[_].kmis)
        size = dfI[dfI['classe']=='C7'].shape[0]
        rd = np.random.randint(size)
        kmis_teste.append(dfI[dfI['classe']=='C7'].iloc[rd].kmis_b14)

    inicio_teste = time.time()
    num_de_instancias = len(kmis_teste)-1
    print(kmis_teste[0])
    acumulado = {
      'HG': {'val': [], 'tempo': []}, 'KI' : {'val': [], 'tempo': []},
      'LS': {'val': [], 'tempo': []}, 'VND': {'val': [], 'tempo': []},
      'TS': {'val': [], 'tempo': []},
      'GRASP_RG_TS' : {'val': [], 'tempo': []},
      'GRASP_RG_VND': {'val': [], 'tempo': []},
      'ANT_TS'  : {'val': [], 'tempo': []},
      'ANT_VND' : {'val': [], 'tempo': []},
      'ANT2_VND': {'val': [], 'tempo': []},
      'ANT2_TS' : {'val': [], 'tempo': []},
    }

    for km in kmis_teste[1:]:
      # if not OLD:
      assert isinstance(km, KMIS)
      tamL, tamR, p, k = km.tamL, km.tamR, km.p, km.k
      heuristicas = {
        # 'HG': (HG_Bogue13, {}),
        # 'KI': (kInterEstendida, {}),
        # 'VND':(VND, {'L_entrada': kInterEstendida(km)}),
        # 'GRASP_RG_VND'      : (GRASP_RG_VND, {'alpha': 0.5,                     'maxIter': 500, 't_lim': 0.5}),
        'GRASP_RG_TS'       : (GRASP_RG_TS,  {'alpha': 0.5,'tau':0.5, 'gama':5, 'maxIter': 1000, 't_lim': 1000}),
        # 'ANT_TS'  : (ANT_TS,   {'alpha':1.0, 'beta':0.8, 'rho':0.3, 'q_zero':0.6, 'qtd_formigas': int(tamL*0.1), 'Q_reativo': 1,  'tau': 0.5, 'gama':5, 'maxIter': 300, 't_lim': 1000}),
        'ANT_VND' : (ANT_VND,  {'alpha':1.0, 'beta':0.8, 'rho':0.3, 'q_zero':1, 'qtd_formigas': int(tamL*0.1), 'Q_reativo': 5,                        'maxIter': 90, 't_lim': 10000}),
        # 'ANT2_VND': (ANT2_VND, {'alpha':1.0, 'beta':0.6, 'rho':0.1, 'q_zero':0.95, 'qtd_formigas_p': 0.1,                                                'maxIter': 100, 't_lim': 10}),
        # 'ANT2_TS' : (ANT2_TS,  {'alpha':1.0, 'beta':0.8, 'rho':0.3, 'q_zero':0.6, 'qtd_formigas_p': 0.1,                          'tau': 0.5, 'gama':5, 'maxIter': 500, 't_lim': 10})
      }
      for rep in range(reps):
        for label, (H, args) in heuristicas.items():
          # if (label == 'KI' or label == 'HG') and rep > 0:
          #   acumulado[label]['val'].append(acumulado[label]['val'][0])
          #   acumulado[label]['tempo'].append(acumulado[label]['tempo'][0])
          #   continue
          args = args.copy()
          valor, tempo, sol, = run(km, H, args)
          acumulado[label]['val'].append(valor)
          acumulado[label]['tempo'].append(tempo)
          if((len(set(sol.L_linha)) < len(sol.L_linha)) or (len(sol)!=km.k)):
            sol.L_linha.sort()
            print(f'Bug! -> {sol}')
          # if label in ["ANT_VND", "ANT2_VND"]:
          print(f"{label:<20} = |∩Si| {valor:<4} | ⌚ {tempo:.4f} | |L'| {len(sol):<4} | kmis {km}")

    print(f"\n 📊 Médias após {reps*num_de_instancias} execuções ({num_de_instancias} instâncias × {reps} reps):")
    for label in acumulado:
      if len(acumulado[label]['val']) > 0:
        valores = np.array(acumulado[label]['val'])
        tempos = np.array(acumulado[label]['tempo'])
        print(f"{label:<14} → |∩Si| = {valores.mean():.4f} ({valores.std(ddof=1):.4f})", end=' ')
        print(f"| ⌚ = {tempos.mean():.6f} ({tempos.std(ddof=1):.4f}) seg")

    total_tempo = time.time()-inicio_teste

    print(f'Tempo total do teste: {int(total_tempo/60)} min e {(total_tempo % 60):.4f} seg.')
    input('Aperte qualquer tecla para continuar...')

    
  """## **Parâmetros**"""

  """ **Teste de Parametros**"""

  #=========================================================================================================                    <---------- Teste Parametros
  # ===== Definindo parametros do Teste =======
  tamGrupoTreino = 34  #param {type: "slider", min:1, max:50, step:1}
  numRep_A       = 10  #param {type: "slider", min:2, max:20, step:1}
  t_lim_A        = 10  #param {type: "slider", min:0.5, max:30, step:0.5}
  tempo_save_A   = 300 #param {type: "slider", min:5, max:600, step:5}

  LIMITE_agendamentos  = 40
  nucleos_teste_param = 10 if os.cpu_count() == 12 else 2

  dArgTest = {
    'ANT_VND' :{'alpha': [1], 'beta':[0.6, 0.8, 1.2], 'rho':[0.1, 0.3, 0.5], 'q_zero':[0.7, 0.8, 0.9], 'qtd_formigas':[0.1, 0.2, 0.3], 'Q_reativo':[5],                           'maxIter': [10000], 't_lim': [t_lim_A]},
    'GRASP_RG_TS' :{'alpha': [0.2, 0.5, 0.8], 'tau': [0.1, 0.3, 0.5], 'gama': [5, 10, 15],   'maxIter': [10000], 't_lim': [t_lim_A]},
    'KIEst':{'t_lim':[t_lim_A]},
    'GRASP_RG_VND':{'alpha': [0.2, 0.5, 0.8],                              'maxIter': [100000], 't_lim': [t_lim_A]},
    'ANT_TS'  :{'alpha': [1], 'beta':[0.8], 'rho':[0.3], 'q_zero':[0.2, 0.5, 0.8], 'qtd_formigas':[0.1, 0.2, 0.3], 'Q_reativo':[5], 'tau': [0.5], 'gama':[5], 'maxIter': [100000], 't_lim': [t_lim_A]},
    'ANT2_VND':{'alpha': [1], 'beta':[0.6, 0.8, 1.2], 'rho':[0.1, 0.3, 0.5], 'q_zero':[0.2, 0.5, 0.8], 'qtd_formigas_p':[0.1, 0.2, 0.3],                                             'maxIter': [100000], 't_lim': [t_lim_A]},
    'ANT2_TS' :{'alpha': [1], 'beta':[0.8], 'rho':[0.3], 'q_zero':[0.2, 0.5, 0.8], 'qtd_formigas_p':[0.1, 0.2, 0.3],                   'tau': [0.5], 'gama':[5], 'maxIter': [100000], 't_lim': [t_lim_A]},
  }
  dfAT = pd.DataFrame()
  if get_boolean_input('Iniciar teste de parâmetros?', 'Teste Parametros'):
    dictAT = {'idH':[], 'idArg': [], 'idI':[], 'rep':[], 'val':[], 'time':[]}
    dfAT = pd.DataFrame(columns = list(dictAT.keys())) # DataFrame com info da performance da Heuristica
    try:
      os.makedirs(os.getcwd()+"/saves_AT", exist_ok=True)
    except: print("Erro ao criar a pasta saves_AT!")
    qtdInstancias = dfI[dfI['temSol']].shape[0]
    # O grupoTreino foi fixado depois do sorteio, para poder reler o arquivo e continuar de onde parou, em caso de teste pausado
    # (algo que foi evitado, os testes rodaram em uma unica execução!)
    grupoTreino_AT = ([152, 237, 270, 147, 288, 148, 269, 151, 304, 211, 336, 192, 319, 276, 102, 155, 24,
                      196, 191, 217, 163, 232, 249, 180, 109, 328, 331, 169, 216,  72,  17, 214,  13, 337]) 
    
    # grupoTreino_AT = choice(qtdInstancias, tamGrupoTreino, replace=False)
    # print(grupoTreino_AT); time.sleep(30)
    # grupoTreino_AT.sort()

    realizado_AT = 0
    setFEITOS : set = set()
    time_start = time.time()

    if get_boolean_input('Carregar dados anteriores do teste de parâmetros?', 'Load AT'):
      try:
        dfAT = pd.read_csv(path_P+'teste_parametros.csv')
        print(f'Leitura de teste_parametros.csv ({dfAT.shape[0]} linhas) bem sucedida.')
      except:
        print("Arquivos de teste_parametros não encontrado!")
        assert dfAT.shape[0]>0, 'Arquivo de teste_parametros.csv solicitado, mas não encontrado!'
      
      for _, row in dfAT.iterrows():
        dadosAT = row.to_dict()
        dictAppend(dictAT, dadosAT)
      setFEITOS = set(zip(dictAT['idH'], dictAT['idArg'], dictAT['idI'], dictAT['rep']))
      dfI_temSol = (dfI[dfI['temSol']].reset_index())[['id']]
      idx = dfI_temSol[dfI_temSol['id'].isin(dfAT['idI'].unique())].reset_index()['index'].to_list()
      if(len(set(idx) - set(grupoTreino_AT)) != 0):
        print(f"\tERRO na obtenção dos ids! Grupo treino do arquivo diferente do original")

      teste_consistencia(dfI, dfAT)

    # ========= Rodando o Teste =============
    ArgANT_VND     = dArgTest['ANT_VND']
    ArgGRASP_RG_TS = dArgTest['GRASP_RG_TS']

    productArg_ant   = list(product(*ArgANT_VND.values()))
    productArg_grasp = list(product(*ArgGRASP_RG_TS.values()))
    TOTAL_iters_AT = (tamGrupoTreino * numRep_A *len(productArg_ant)) + (tamGrupoTreino * numRep_A *len(productArg_grasp))
    iters_restantes = TOTAL_iters_AT-len(setFEITOS)
    # Teste de argumentos do ANT_VND
    os.system('cls' if os.name == 'nt' else 'clear')
    print(' -------- Teste de Parâmetros ------------')
    print(f'Tamanho GT: {len(grupoTreino_AT)}, t_lim: {t_lim_A}, rep: {numRep_A}, Tempo saves: {tempo_save_A}'
          +f'\nNucleos: {nucleos_teste_param}, Agendamentos: {LIMITE_agendamentos}, Carregados: {len(setFEITOS)}')
    agendados_AT = set()
    with ProcessPoolExecutor(max_workers = nucleos_teste_param) as executor:
      with tqdm(total=iters_restantes, smoothing = 0.001, desc="Testando argumentos ANT_VND") as pbar:
        for k in dfI[dfI['temSol']].index[grupoTreino_AT]:
          kmis = dfI.loc[k].kmis
          for rep in range(numRep_A):
            for argID in productArg_ant:
              alpha, beta, rho, q_zero, qtd_formigas, Q_reativo, maxIter, t_lim_A = argID
              if(('ANT_VND', str(argID), dfI.loc[k, 'id'], rep) not in setFEITOS):
                arg = {'alpha'         : alpha,
                      'beta'           : beta,
                      'rho'            : rho,
                      'q_zero'         : q_zero,
                      'qtd_formigas'   : min(max(2, int((qtd_formigas*kmis.tamL)+0.5)), kmis.tamL),
                      'Q_reativo'      : Q_reativo,
                      'maxIter'        : 10_000_000, #min(max(10, int(maxIter*kmis.tamL)) , 1000),
                      't_lim'          : t_lim_A*(kmis.tamL/MAX_TAMANHO_L)}
                tarefa_agendada = executor.submit(run_wrapper, 'ANT_VND', str(argID), kmis, arg, k, rep)
                agendados_AT.add(tarefa_agendada)
              else: realizado_AT+=1

              if(len(agendados_AT)>=LIMITE_agendamentos):
                concluidos, agendados_AT = wait(agendados_AT, return_when=FIRST_COMPLETED)
                for tarefa in concluidos:
                  val_a, tempo_a, _, idH_a, argID_a, k_a, rep_r= tarefa.result()

                  dadosAT = { 'idH': idH_a,
                              'idArg': argID_a,
                              'idI': dfI.loc[k_a, 'id'], 'rep':rep_r, 'val':val_a, 'time':tempo_a}

                  dictAppend(dictAT, dadosAT)
                  pbar.update(1)
                  realizado_AT+=1
                if(((time.time()-time_start)>tempo_save_A)  and (realizado_AT % 5 == 0)):
                  time_start = time.time()
                  save_df(dictAT, realizado_AT, 'AT')
                  pbar.set_postfix({"Salvos": f"{realizado_AT} ({(realizado_AT/TOTAL_iters_AT)*100:.2f}%)"})

        if(len(agendados_AT)>0):
          concluidos, agendados_AT = wait(agendados_AT, return_when=ALL_COMPLETED)
          for tarefa in concluidos:
            val_a, tempo_a, _, idH_a, argID_a, k_a, rep_r= tarefa.result()
            dadosAT = { 'idH':idH_a,
                        'idArg': argID_a,
                        'idI': dfI.loc[k_a, 'id'], 'rep':rep_r, 'val':val_a, 'time':tempo_a}
            dictAppend(dictAT, dadosAT)
            pbar.update(1)
            realizado_AT+=1
          time_start = time.time()
          save_df(dictAT, realizado_AT, 'AT')
          pbar.set_postfix({"Salvos": f"{realizado_AT} ({(realizado_AT/TOTAL_iters_AT)*100:.2f}%)"})

        pbar.set_description("Testando argumentos GRASP_RG_TS")
        #Teste de argumentos do GRASP
        for k in dfI[dfI['temSol']].index[grupoTreino_AT]:
          kmis = dfI.loc[k].kmis
          for rep in range(numRep_A):
            for argID in productArg_grasp:
              alpha, tau, gama, maxIter, t_lim_A = argID
              if(('GRASP_RG_TS', str(argID), dfI.loc[k, 'id'], rep) not in setFEITOS):
                arg = {'alpha'         : alpha,
                      'tau'            : tau,
                      'gama'           : gama,
                      'maxIter'        : 10_000_000,  #min(max(10, int(maxIter*kmis.tamL)) , 1000),
                      't_lim'          : t_lim_A*(kmis.tamL/MAX_TAMANHO_L)}
                tarefa_agendada = executor.submit(run_wrapper, 'GRASP_RG_TS', str(argID), kmis , arg, k, rep)
                agendados_AT.add(tarefa_agendada)
              else: realizado_AT+=1
              if(len(agendados_AT)>=LIMITE_agendamentos):
                concluidos, agendados_AT = wait(agendados_AT, return_when=FIRST_COMPLETED)
                for tarefa in concluidos:
                  val_a, tempo_a, _, idH_a, argID_a, k_a, rep_a = tarefa.result()
                  dadosAT = {'idH' :idH_a,
                             'idArg': argID_a,
                             'idI':dfI.loc[k_a, 'id'], 'rep':rep_a, 'val':val_a, 'time':tempo_a}

                  dictAppend(dictAT, dadosAT)
                  pbar.update(1)
                  realizado_AT+=1
                if(((time.time()-time_start)>tempo_save_A) and (realizado_AT % 5 == 0)):
                  time_start = time.time()
                  save_df(dictAT, realizado_AT, 'AT')
                  pbar.set_postfix({"Salvos": f"{realizado_AT} ({(realizado_AT/TOTAL_iters_AT)*100:.2f}%)"})

        if(len(agendados_AT)>0):
          concluidos, agendados_AT = wait(agendados_AT, return_when=ALL_COMPLETED)
          for tarefa in concluidos:
            val_a, tempo_a, _, idH_a, argID_a, k_a, rep_a= tarefa.result()
            dadosAT = {'idH' :idH_a,
                       'idArg': argID_a,
                       'idI':dfI.loc[k_a, 'id'], 'rep':rep_a, 'val':val_a, 'time':tempo_a}
            dictAppend(dictAT, dadosAT)
            pbar.update(1)
            realizado_AT+=1
          time_start = time.time()
          save_df(dictAT, realizado_AT, 'AT')
          pbar.set_postfix({"Salvos": f"{realizado_AT} ({(realizado_AT/TOTAL_iters_AT)*100:.2f}%)"})
        
    dfAT = pd.DataFrame(dictAT)
    dfAT.to_csv(path_P+'teste_parametros.csv', index=False)
  else:
    try:
      dfAT = pd.read_csv(path_P+'teste_parametros.csv')
      print(f'Leitura de teste_parametros.csv ({dfAT.shape[0]} linhas) bem sucedida.')
    except:
      print(f'Arquivo de teste_parametros não encontrado!')
      assert dfAT.shape[0]>0, 'Arquivo teste_parametros não encontrado!'

  # ==================== Formatando e obtendo as estatisticas desejadas ================
  assert isinstance(dfAT, pd.DataFrame), '\t ⚠️ DataFrame dfAT não definido.'
  df_ai  = dfAT.groupby(['idH', 'idArg', 'idI'])[['val', 'time']].apply(junta_repeticoes).reset_index()
  df_a   = df_ai.groupby(['idH', 'idArg'])[['vmin', 'vavg', 'vmax', 'tavg']].apply(medias).reset_index()
  df_i   = df_ai.groupby(['idH', 'idI'])[['vmin', 'vavg', 'vmax', 'tavg']].apply(melhor_por_instancia).reset_index()

  # Comparações
  cmp            = df_ai.merge(df_i, on=['idH', 'idI'])
  cmp['eq_vmin'] = cmp['vmin']  ==  cmp['vmin_max']
  cmp['eq_vmax'] = cmp['vmax']  ==  cmp['vmax_max']
  cmp['eq_vavg'] = np.isclose(cmp['vavg'], cmp['vavg_max'], atol=1e-4)
  cmp['eq_tavg'] = np.isclose(cmp['tavg'], cmp['tavg_min'], atol=1e-4)

  # Contagem e merge  final
  cnt = cmp.groupby(['idH', 'idArg'])[['eq_vmin', 'eq_vavg', 'eq_vmax', 'eq_tavg']].sum().rename(columns=lambda c: 'cnt_' + c[3:]).reset_index()
  dfA = df_a.merge(cnt, on=['idH', 'idArg'])

  df_limites   = dfA.groupby(['idH'])[dfA.columns[2:]].apply(limites_argumento).reset_index()
  df_score     = dfA.merge(df_limites, on=['idH'])
  dfA['score']  = df_score.apply(score_time_off, axis=1)

  teste_consistencia(dfI, dfAT)

  """**TOP 5 Parametros**"""
  #=====================================================================================
  top5 = dfA.groupby(['idH'])[dfA.columns[1:]].apply(lambda g: g.nlargest(5, 'score'))
  top5.to_csv(path_P+'top5_teste_parametros.csv')

  print(top5)


  """Função de parametros estática"""
  #===================================================================================
  def dArg(tamL, k, t_lim, idH)->dict:

    argANT_VND = [1, 1.2, 0.1, 0.9, 0.1, 5, 10000, 10]

    argGRG_TS = [0.8, 0.3, 5, 10000, 10]

    if(idH == 'ANT_VND2')     : idH = 'ANT_VND'
    if(idH == 'ANT2_VND2')    : idH = 'ANT2_VND'
    if(idH == 'GRASP_RG_VND2'): idH = 'GRASP_RG_VND'

    dArgs ={'KIEst':{'t_lim':t_lim},
          'GRASP_RG_TS':{'alpha'  : argGRG_TS[0],
                        'tau'    : argGRG_TS[1],
                        'gama'   : argGRG_TS[2],
                        'maxIter': 10_000_000, #min(max(50, int(argGRG_TS[3]*tamL)) , 10000),
                        't_lim'  : t_lim*(tamL/MAX_TAMANHO_L)},

          'GRASP_RG_VND':{'alpha'  : argGRG_TS[0],
                          'maxIter': 10_000_000, #min(max(50, int(argGRG_TS[3]*tamL)) , 10000),
                          't_lim'  : t_lim*(tamL/MAX_TAMANHO_L)},

          'ANT_VND':{'alpha'       : argANT_VND[0],
                    'beta'        : argANT_VND[1],
                    'rho'         : argANT_VND[2],
                    'q_zero'      : argANT_VND[3],
                    'qtd_formigas': min(max(2, int((argANT_VND[4]*tamL)+0.5)), tamL),
                    'Q_reativo'   : argANT_VND[5],
                    'maxIter'     : 10_000_000, #min(max(50, int(argANT_VND[6]*tamL)) , 10000),
                    't_lim'       : t_lim*(tamL/MAX_TAMANHO_L)},

          'ANT_TS':{ 'alpha'       : argANT_VND[0],
                    'beta'        : argANT_VND[1],
                    'rho'         : argANT_VND[2],
                    'q_zero'      : argANT_VND[3],
                    'qtd_formigas': min(max(2, int((argANT_VND[4]*tamL)+0.5)), tamL),
                    'Q_reativo'   : argANT_VND[5],
                    'tau'         : argGRG_TS[1],
                    'gama'        : argGRG_TS[2],
                    'maxIter'     : 10_000_000, #min(max(50, int(argANT_VND[6]*tamL)) , 10000),
                    't_lim'       : t_lim*(tamL/MAX_TAMANHO_L)},

          'ANT2_VND':{ 'alpha'         : argANT_VND[0],
                      'beta'          : argANT_VND[1],
                      'rho'           : argANT_VND[2],
                      'q_zero'        : argANT_VND[3],
                      'qtd_formigas_p': argANT_VND[4],
                      'maxIter'       : 10_000_000, #min(max(50, int(argANT_VND[6]*tamL)) , 10000),
                      't_lim'         : t_lim*(tamL/MAX_TAMANHO_L)},

          'ANT2_TS':{'alpha'         : argANT_VND[0],
                    'beta'          : argANT_VND[1],
                    'rho'           : argANT_VND[2],
                    'q_zero'        : argANT_VND[3],
                    'qtd_formigas_p': argANT_VND[4],
                    'tau'           : argGRG_TS[1],
                    'gama'          : argGRG_TS[2],
                    'maxIter'       : 10_000_000, #min(max(50, int(argANT_VND[6]*tamL)) , 10000),
                    't_lim'         : t_lim*(tamL/MAX_TAMANHO_L)},

          }

    return dArgs[idH]

  """## **Resultados**"""

  """Rodando Heuristicas"""
  # =====================================================================================                             <---------------- Rodando Heuristicas
  num_rep_R    = 10  
  t_lim_R      = 10 
  tempo_save_R = 300
  nucleos_teste_final  = 10 if os.cpu_count() == 12 else 2
  LIMITE_agendamentos_R = 40
  H_teste = ['KIEst'  , 'GRASP_RG_TS', 'GRASP_RG_VND', 'GRASP_RG_VND2', 'ANT_TS',
              'ANT_VND', 'ANT_VND2'   , 'ANT2_TS'     , 'ANT2_VND'    , 'ANT2_VND2']
  dfRT = pd.DataFrame()
  if get_boolean_input('Iniciar o teste de heurísticas final?', 'Teste Final'):
    dictRT = {'idH':[], 'idI':[], 'rep':[], 'val':[], 'time':[]}#, 'val_b14':[], 'time_b14':[]}
    dfRT = pd.DataFrame(columns = list(dictRT.keys())) # DataFrame com info da performance da Heuristica
    try:
      os.makedirs(os.getcwd()+"/saves_RT", exist_ok=True)
    except: print("Erro ao criar a pasta saves_RT!")
    qtdInstancias = dfI[dfI['temSol']].shape[0]
    OrdemTeste = np.random.permutation(qtdInstancias) # Ordem randomica para melhor previsão de termino

    realizado_R = 0
    setFEITOS_R = set()

    if get_boolean_input('Carregar dados anteriores do teste final?', 'Load RT'):
      conv = {'sol' : ast.literal_eval, 'sol_b14' : ast.literal_eval}
      try:
        dfRT = pd.read_csv(path_P+'resultados.csv', converters=conv)
        print(f'Leitura de resultados.csv ({dfRT.shape[0]} linhas) bem sucedida.')
      except:
        print(f'Arquivo de resultados não encontrado!')
        assert dfRT.shape[0]>0, 'Sem arquivo resultados.csv, que foi solicitado.'

      for _, row in dfRT.iterrows():
        dadosRT = row.to_dict()
        dictAppend(dictRT, dadosRT)
      setFEITOS_R = set(zip(dictRT['idH'], dictRT['idI'], dictRT['rep']))
      teste_consistencia(dfI, dfRT)

    total_iters_R = qtdInstancias * num_rep_R * len(H_teste)
    iters_restante_R = total_iters_R - len(setFEITOS_R)

    time_start_R = time.time()
    os.system('cls' if os.name == 'nt' else 'clear')
    print(' -------- Teste nas Instâncias Base ------------')
    print(f't_lim: {t_lim_R}, rep: {num_rep_R} Tempo saves: {tempo_save_R}'+
          f'\nNucleos:{nucleos_teste_final}, Agendamentos: {LIMITE_agendamentos_R}')
    agendados_R = set()
    with ProcessPoolExecutor(max_workers=nucleos_teste_final) as executor:
      with tqdm(total=iters_restante_R, smoothing = 0.001, desc="Executando heurísticas") as pbar:
        for k in dfI[dfI['temSol']].index[OrdemTeste]:
          kmis = dfI.loc[k].kmis
          for rep in range(num_rep_R):  #numero de repetições
            for H in H_teste:
              if (H, dfI.loc[k, 'id'], rep) not in setFEITOS_R:
                tarefa_agendada = executor.submit(run_wrapper, H, '_', kmis, dArg(kmis.tamL, kmis.k, t_lim_R, H), k, rep)
                agendados_R.add(tarefa_agendada)
              else:
                realizado_R+=1
              if(len(agendados_R)>=LIMITE_agendamentos_R):
                concluidos, agendados_R = wait(agendados_R, return_when=FIRST_COMPLETED)
                for feito in concluidos:
                  val_RT, tempo_RT, _,H_RT, _, k_RT, rep_RT = feito.result()
                  dadosIte = {'idH':f'{H_RT}', 'idI':dfI.loc[k_RT, 'id'], 'rep':f"{rep_RT}",  'val':val_RT, 'time':tempo_RT }
                  dictAppend(dictRT, dadosIte)
                  pbar.update(1)
                  realizado_R+=1
                if(((time.time()-time_start_R)>tempo_save_R) and (realizado_R % 5 == 0)):
                  time_start_R = time.time()
                  save_df(dictRT, realizado_R, 'RT')
                  pbar.set_postfix({"Salvos": f"{realizado_R} ({(realizado_R/total_iters_R)*100:.2f}%)"})

        if(len(agendados_R)>0):
          concluidos, agendados_R = wait(agendados_R, return_when=ALL_COMPLETED)
          for feito in concluidos:
            val_RT, tempo_RT, _, H_RT, _, k_RT, rep_RT = feito.result()
            dadosIte = {'idH':f'{H_RT}', 'idI':dfI.loc[k_RT, 'id'],'rep':rep_RT, 'val':val_RT, 'time':tempo_RT }
            dictAppend(dictRT, dadosIte)
            pbar.update(1)
            realizado_R+=1

          time_start_R = time.time()
          save_df(dictRT, realizado_R, 'RT')
          pbar.set_postfix({"Salvos": f"{realizado_R} ({(realizado_R/total_iters_R)*100:.2f}%)"})

    dfRT = pd.DataFrame(dictRT)
    dfRT.to_csv(path_P+'resultados.csv', index=False)
  else:
    conv = {'sol' : ast.literal_eval, 'sol_b14' : ast.literal_eval}
    try:
      dfRT = pd.read_csv(path_P+'resultados.csv', converters=conv)
      print(f'Leitura de resultados.csv ({dfRT.shape[0]} linhas) bem sucedida.')
    except:
      print(f'Arquivo de resultados não encontrado!')
      assert dfRT.shape[0] > 0, 'Erro arquivo faltante ou vazio!'  

  assert isinstance(dfRT, pd.DataFrame), '\t ⚠️ DataFrame dfRT não definido.'
  teste_consistencia(dfI, dfRT)



  """Rodando Heuristicas nas Instâncias Reduzidas!"""
  # =====================================================================================                  <---------------- Rodando Heuristicas Instâncias Reduzidas
  num_rep_IRT    = 10  
  t_lim_IRT      = 10 
  tempo_save_IRT = 300
  nucleos_teste_IRT  = 10 if os.cpu_count() == 12 else 2
  LIMITE_agendamentos_IRT = 40
  H_TESTE_IR : list[str] = ['KIEst'  , 'GRASP_RG_TS', 'GRASP_RG_VND', 'GRASP_RG_VND2', 'ANT_TS',
           'ANT_VND', 'ANT_VND2'   , 'ANT2_TS'     , 'ANT2_VND'    , 'ANT2_VND2']
  dfIRT = pd.DataFrame()
  if get_boolean_input('Iniciar o teste de heurísticas nas instâncias reduzidas?', 'Teste Final'):
    dictT_IRT = {'idH':[], 'idI':[], 'rep':[], 'val':[], 'time':[]}#, 'val_b14':[], 'time_b14':[]}
    dfIRT = pd.DataFrame(columns = list(dictT_IRT.keys())) # DataFrame com info da performance da Heuristica
    try:
      os.makedirs(os.getcwd()+"/saves_IRT", exist_ok=True)
    except: print("Erro ao criar a pasta saves_IRT!")

    qtdInstancias_IRT = dfI[dfI['temSol']].shape[0]
    OrdemTeste_IRT = np.random.permutation(qtdInstancias_IRT) # Ordem randomica para melhor previsão de termino

    realizado_IRT = 0
    setFEITOS_IRT = set()

    if get_boolean_input('Carregar dados anteriores do teste nas instâncias reduzidas?', 'Load RT'):
      conv = {'sol' : ast.literal_eval, 'sol_b14' : ast.literal_eval}
      try:
        dfIRT = pd.read_csv(path_P+'resultados_reduzidas.csv', converters=conv)
        print(f'Leitura de resultados_reduzidas.csv ({dfIRT.shape[0]} linhas) bem sucedida.')
      except:
        print(f'Arquivo de resultados_reduzidas não encontrado!')
        assert dfIRT.shape[0]>0, 'Sem arquivo resultados_reduzidas.csv, que foi solicitado.'

      for _, row in dfIRT.iterrows():
        dadosRT = row.to_dict()
        dictAppend(dictT_IRT, dadosRT)
      setFEITOS_IRT = set(zip(dictT_IRT['idH'], dictT_IRT['idI'], dictT_IRT['rep']))
      teste_consistencia(dfI, dfIRT)

    total_iters_IRT = qtdInstancias_IRT * num_rep_IRT * len(H_TESTE_IR)
    iters_restante_IRT = total_iters_IRT - len(setFEITOS_IRT)

    time_start_IRT = time.time()
    os.system('cls' if os.name == 'nt' else 'clear')
    print(' -------- Teste nas Instâncias Reduzidas ------------')
    print(f't_lim: {t_lim_IRT}, rep: {num_rep_IRT} Tempo saves: {tempo_save_IRT}'+
          f'\nNucleos:{nucleos_teste_IRT}, Agendamentos: {LIMITE_agendamentos_IRT}')
    agendados_IRT = set()
    with ProcessPoolExecutor(max_workers=nucleos_teste_IRT) as executor:
      with tqdm(total=iters_restante_IRT, smoothing = 0.001, desc="Executando heurísticas") as pbar:
        for k in dfI[dfI['temSol']].index[OrdemTeste_IRT]:
          kmis_b14 = dfI.loc[k].kmis_b14
          for rep in range(num_rep_IRT):  #numero de repetições
            for H in H_TESTE_IR:
              if (H, dfI.loc[k, 'id'], rep) not in setFEITOS_IRT:
                tarefa_agendada_IRT = executor.submit(run_wrapper, H, '_', kmis_b14, dArg(kmis_b14.tamL, kmis_b14.k, t_lim_IRT, H), k, rep)
                agendados_IRT.add(tarefa_agendada_IRT)
              else:
                realizado_IRT+=1
              if(len(agendados_IRT)>=LIMITE_agendamentos_IRT):
                concluidos, agendados_IRT = wait(agendados_IRT, return_when=FIRST_COMPLETED)
                for feito in concluidos:
                  val_IRT, tempo_IRT, _,H_IRT, _, k_IRT, rep_IRT = feito.result()
                  dadosIte = {'idH':f'{H_IRT}', 'idI':dfI.loc[k_IRT, 'id'], 'rep':f"{rep_IRT}",  'val':val_IRT, 'time':tempo_IRT }
                  dictAppend(dictT_IRT, dadosIte)
                  pbar.update(1)
                  realizado_IRT+=1
                if(((time.time()-time_start_IRT)>tempo_save_IRT) and (realizado_IRT % 5 == 0)):
                  time_start_IRT = time.time()
                  save_df(dictT_IRT, realizado_IRT, 'IRT')
                  pbar.set_postfix({"Salvos": f"{realizado_IRT} ({(realizado_IRT/total_iters_IRT)*100:.2f}%)"})

        if(len(agendados_IRT)>0):
          concluidos, agendados_IRT = wait(agendados_IRT, return_when=ALL_COMPLETED)
          for feito in concluidos:
            val_IRT, tempo_IRT, _, H_IRT, _, k_IRT, rep_IRT = feito.result()
            dadosIte = {'idH':f'{H_IRT}', 'idI':dfI.loc[k_IRT, 'id'],'rep':rep_IRT, 'val':val_IRT, 'time':tempo_IRT }
            dictAppend(dictT_IRT, dadosIte)
            pbar.update(1)
            realizado_IRT+=1

          time_start_IRT = time.time()
          save_df(dictT_IRT, realizado_IRT, 'IRT')
          pbar.set_postfix({"Salvos": f"{realizado_IRT} ({(realizado_IRT/total_iters_IRT)*100:.2f}%)"})

    dfIRT = pd.DataFrame(dictT_IRT)
    dfIRT.to_csv(path_P+'resultados_reduzidas.csv', index=False)
  else:
    conv = {'sol' : ast.literal_eval, 'sol_b14' : ast.literal_eval}
    try:
      dfIRT = pd.read_csv(path_P+'resultados_reduzidas.csv', converters=conv)
      print(f'Leitura de resultados_reduzidas.csv ({dfIRT.shape[0]} linhas) bem sucedida.')
    except:
      print(f'Arquivo de resultados_reduzidas não encontrado!')
      assert dfIRT.shape[0] > 0, 'Erro arquivo faltante ou vazio!'  

  assert isinstance(dfIRT, pd.DataFrame), '\t ⚠️ DataFrame dfIRT não definido.'
  teste_consistencia(dfI, dfIRT)

if __name__=="__main__":
  main()