from linear_algebra import dot
import math

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def neuronio_MCP(pesos, entradas):
    uk = dot(pesos, entradas)
    return sigmoid(uk)

def feed_forward(rede_xor, vetor_entrada):
    """Recebe uma rede neura (representada como uma lista de listas de listas de pesos)
    e retorna a saída da propagação direta da entrada"""
    vetor_saida = []
    
    for ponteiro in rede_xor:
        
        entrada_com_bias = vetor_entrada + [1]
        
        saida = [neuronio_MCP(neuronio, entrada_com_bias)
                 for neuronio in ponteiro]
        vetor_saida.append(saida)
        
        vetor_entrada = saida
        # a entrada apara a príxima camada é a saída desta etapa
    return vetor_saida

# função não mais útil
def degrau(x): 
    return 1 if x >= 0 else 0

sinapses_xor = [# camada oculta
                [[20, 20, -30],
                 [20, 20, -10]],
                # camada de saída
                [[-60, 60, -30]]]

for x in [0, 1]:
    for y in [0, 1] :
        print (x, " EXCLUSIVO ", y, " = ", feed_forward(sinapses_xor, [x, y])[-1])