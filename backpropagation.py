from linear_algebra import dot
import math

def sigmoid(t):
    return ((2 / (1 + math.exp(-t)))-1)

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

alpha = 0.08
def backpropagation(rede_neural, vetor_entrada, vetor_saida):
    saidas_intermediarias, saidas_neuronios = feed_forward(rede_neural, vetor_entrada)
    deltas_saida = [(0.5 * (1 + saida)* (1 - saida)) * (saida - vetor_saida[i]) * alpha
                    for i, saida in enumerate(saidas_neuronios)]
    
    for i, neuronio_saida, in enumerate(rede_neural[-1]):
        for j, saida_intermediaria in enumerate(saidas_intermediarias + [1]):
            neuronio_saida[j] -= deltas_saida[i] * saida_intermediaria
    
    deltas_intermediarios = [0.5 * alpha * (1 + saida_intermediaria) * (1 - saida_intermediaria) *
                            dot(deltas_saida, [n[i] for n in rede_neural[-1]])
                            for i, saida_intermediaria in enumerate(saidas_intermediarias)]
    
    for i, neuronio_intermediario in enumerate(rede_neural[0]):
        for j, input in enumerate(vetor_entrada + [1]):
            neuronio_intermediario[j] -= deltas_intermediarios[i] * input

hidden_layer = [[-0.1, -1.1, 1.55], [-0.91, -0.81, 0.125]]
output_layer = [[0.2, -2.1, 0.98]]

e_network = [hidden_layer, output_layer]

print("e_network = ", e_network, "\n")

inputs = [[0,0], [0,1], [1,0], [1,1]]
targets = [[1], [0], [0], [1]]

# TREINAMENTO DA REDE NEURAL

for n in range(10000): # numero de ciclos de treinamento
    for input_vetor, target_vetor in zip(inputs, targets):
        backpropagation(e_network, input_vetor, target_vetor)

print("pesos sinápticos atualizados = ", e_network, "\n")
# tede de rede neural inserindo a tebela de entrada nos pesos sinapticos do final do treinamento
for x in [0, 1]:
    for y in [0, 1]:
        print(x, " COINCIDENCIA ", y, " = ", feed_forward(e_network, [x, y])[-1])