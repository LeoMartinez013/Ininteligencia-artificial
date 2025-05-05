from collections import Counter
from functools import partial
from linear_algebra import dot
import math, random
import matplotlib
import matplotlib.pyplot as plt

def step_function(x):
    return 1 if x >= 0 else 0

def perceptron_output(weights, bias, x):
    """incova função degrau = step_function"""
    return step_function(dot(weights, x) + bias)

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))

def feed_forward(rede_xor, vetor_entrada):
    """Recebe uma rede neura (representada como uma lista de listas de listas de pesos)
    e retorna a saída da propagação direta da entrada"""
    vetor_saida = []
    
    for ponteiro in rede_xor:
        
        entrada_com_bias = vetor_entrada + [1]
        saida = [neuron_output(neuronio, entrada_com_bias)
                 for neuronio in ponteiro]
        vetor_saida.append(saida)
        
        # a entrada apara a príxima camada é a saída desta etapa
        vetor_entrada = saida
    return vetor_saida

alpha = 0.08

def backpropagation(network, input_vector, target):
    hidden_outputs, outputs = feed_forward(network, input_vector)
    
    output_deltas = [(0.5 * (1 + saida)* (1 - saida)) * (saida - target[i]) * alpha
                    for i, saida in enumerate(outputs)]
    
    for i, output_neuron, in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output
    
    hidden_deltas = [0.5 * alpha * (1 + hidden_output) * (1 - hidden_output) *
                            dot(output_deltas, [n[i] for n in network[-1]])
                            for i, hidden_output in enumerate(hidden_outputs)]
    
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input

if __name__ == "__main__":
    raw_digits = [
            # 0
            """11111
               1...1
               1...1
               1...1
               11111""",
            # 1
            """..1..
               ..1..
               ..1..
               ..1..
               ..1..""",
            # 2
            """11111
               ....1
               11111
               1....
               11111""",
            # 3
            """11111
               ....1
               11111
               ....1
               11111""",
            # 4
            """1...1
               1...1
               11111
               ....1
               ....1""",
            # 5
            """11111
               1....
               11111
               ....1
               11111""",
            # 6
            """11111
               1....
               11111
               1...1
               11111""",
            # 7
            """11111
               ....1
               ....1
               ....1
               ....1""",
            # 8
            """11111
               1...1
               11111
               1...1
               11111""",
            # 9 
            """11111
               1...1
               11111
               ....1
               ....1"""
    ]
    
    def make_digit(raw_digit):
        return [1 if c == '1' else 0
                for row in raw_digit.split("\n")
                for c in row.split()]

    inputs = list(map(make_digit, raw_digits))

    targets = [[1 if i == j else 0 for i in range(10)]
            for j in range(10)]

    random.seed(0)      # pode-se utilizar valores repetidos a partir dos randômicos
    input_size = 25     # dimenswão dos vetores relacionados as 10 entradas
    num_hidden = 5      # quantidade de neurônios na camada intermediária
    output_size = 10    # 10 saidas, cada uma relacioanda a uma entrada

    hidden_layer = [[random.random() for __ in range(input_size + 1)]
                    for __ in range(num_hidden)]

    output_layer = [[random.random() for __ in range(num_hidden + 1)]
                    for __ in range(output_size)]

    network = [hidden_layer, output_layer]

    # 10000 CICLOS DE TREINAMENTO
    for __ in range(10000):
        for input_vector, target_vector in zip(inputs, targets):
            print("DEBUG input_vector", input_vector)
            print("DEBUG target_vector", target_vector)
            backpropagation(network, input_vector, target_vector)
    
    def predict(input):
        return feed_forward(network, input)[-1]

# TESTE DAS ENTRADAS TREINADAS
    for i, input in enumerate(inputs):
        outputs = predict(input)
        print(i, [round(p, 2) for p in outputs]) # resultado dos neurônios de saida com 2 casas decimais 

# TESTE DE DIGITOS NUMÉRICOS QUE NÃO FORMA TREINADOS
    print(""".@@@.
...@@
..@@.
...@@
.@@@.""")
    print([round(x, 2) for x in
           predict( [0,1,1,1,0,
                     0,0,0,1,1,
                     0,0,1,1,0,
                     0,0,0,1,1,
                     0,1,1,1,0])])
    print("interpreta com digito 3 com vestígios do digito 9")
    print(""".@@@.
@..@@
.@@@.
@..@@
.@@@.""")
    print([round(x, 2) for x in
           predict( [0,1,1,1,0,
                     1,0,0,1,1,
                     0,1,1,1,0,
                     1,0,0,1,1,
                     0,1,1,1,0])])
    print("interpreta como possível dígito 5, 8 ou 9")
    