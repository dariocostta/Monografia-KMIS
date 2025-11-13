# Monografia-KMIS
Arquivos da pesquisa experimental de heurísticas aplicadas ao problema de Maxima Interseção de k-Subconjuntos.

## Instâncias
As instancias são armazenadas em "arquivos_principais/instancias.csv", cada linha representando uma instancia original e em sequida com a redução de Bogue 2014 aplicada.

Cada linha contem:

| <div align="center">Tipo</div> | <div align="center">Coluna</div> | Descrição |
|:------------------------------:|:-------------------------------:|-----------|
| <span style="color:#D4A373">str</span>       | `id`             | Identificador único da instância, exemplo `"C1p29k7L40R32_0"`. |
| <span style="color:#6C91BF">float</span>     | `p`              | Densidade de arestas. |
| <span style="color:#7CA982">int</span>       | `k`              | Número de subconjuntos a ser escolhido. |
| <span style="color:#7CA982">int</span>       | `\|L\|`          | Tamanho do conjunto `L`. |
| <span style="color:#7CA982">int</span>       | `\|R\|`          | Tamanho do conjunto `R`. |
| <span style="color:#A78FC4">list[int]</span> | `L`              | Lista de inteiros cujos bits representam cada subconjunto  $S_i$. |
| <span style="color:#C97C7C">bool</span>      | `temSol`         | Indica se a instância possui solução (`True` ou `False`). |
| <span style="color:#D4A373">str</span>       | `classe`         | Classe conforme Bogue 2013. |
| <span style="color:#7CA982">int</span>       | `\|L\|_b14`      | Tamanho do conjunto `L` após redução com base no método `bogue14`. |
| <span style="color:#7CA982">int</span>       | `\|R\|_b14`      | Tamanho do conjunto `R` após redução com base no método `b14`. |
| <span style="color:#A78FC4">list[int]</span> | `L_b14`          | Lista de inteiros após aplicação do método de redução `b14`. |
| <span style="color:#6C91BF">float</span>     | `tempo_reducao`  | Tempo (em segundos) necessário para realizar a redução da instância. |
| <span style="color:#A78FC4">list[int]</span> | `Llabel_b14`     | Lista de rótulos associados aos elementos de `L_b14`. |
| <span style="color:#A78FC4">list[int]</span> | `Rlabel_b14`     | Lista de rótulos associados aos elementos de `R_b14`. |
| <span style="color:#D4A373">str</span>       | `classe_b14`     | Classe atribuída à instância após aplicação da redução `b14`. |
| <span style="color:#6C91BF">float</span>     | `p_b14`          | Valor da densidade `p` recalculada após a redução `b14`. |

> Recomendo dar cast nos tipos corretos no momento de reinstanciar, pois a pandas costuma trocar o tipo `int` por `numpy.int64` e isso pode dar overflow.

## Principais Heuristicas implementadas
- "Principais" - Chamadas diretamente nos testes
  - Greedy Randomized Adaptive Search Procedure (GRASP)
  - Colônia de FOrmigas (ANT)
- Intersificações
  - Busca Tabu (TS)
  - Variable Neighborgood Descent (VND)

Todas em `KMIS\bibkmis\heuristicaskmis.py`. 