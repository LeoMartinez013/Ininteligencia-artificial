from typing import List
import matplotlib.pyplot as plt
import numpy as np

Vector = List[float]

def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w)
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def degrau(x):
    return 1 if x >= 0 else 0

def perceptron_output(pesos, entradas):
    y = dot(pesos, entradas)
    return degrau(y)

def ajustes(sinapses, entradas, saida):
    taxa_aprendizagem = 0.5
    saida_parcial = perceptron_output(sinapses, entradas)
    
    for j in range(3):
        sinapses[j] = sinapses[j] + taxa_aprendizagem * (saida[0] - saida_parcial) * entradas[j]
    
    saida = saida_parcial
    return sinapses, saida

def teste_generalizacao(sinapses, entradas, saida):
    saida_parcial = perceptron_output(sinapses, entradas)
    saida = saida_parcial
    return sinapses, saida

neuronio = [0.5, -0.2, 0.3]

# Dados de treinamento: [viés, Poder, Agilidade]
# Contexto Marvel Rivals:
# - Jogadores com performance baixa (esperados como Perdedores) -> Classe 0
# - Jogadores com performance alta (esperados como Vencedores) -> Classe 1

# Jogadores Perdedores (Classe 0)
padrao0 = [-1, 10, 15]   # Jogador com baixo poder e agilidade
padrao1 = [-1, 20, 18]
padrao2 = [-1, 15, 20]
padrao3 = [-1, 18, 17]
padrao4 = [-1, 22, 16]
padrao5 = [-1, 19, 19]

# Jogadores Vencedores (Classe 1)
padrao6  = [-1, 80, 25]  # Jogador com alto poder e agilidade
padrao7  = [-1, 75, 30]
padrao8  = [-1, 85, 28]
padrao9  = [-1, 90, 35]
padrao10 = [-1, 88, 32]
padrao11 = [-1, 92, 31]

saida0 = [0]  # Classe 0 - Perdedores
saida1 = [1]  # Classe 1 - Vencedores

n = 0
for _ in range(15):
    # Treina com jogadores Perdedores (Classe 0)
    neuronio, saida_0 = ajustes(neuronio, padrao0, saida0)
    print(neuronio, "saida0 =", saida_0)
    neuronio, saida_0 = ajustes(neuronio, padrao1, saida0)
    print(neuronio, "saida0 =", saida_0)
    neuronio, saida_0 = ajustes(neuronio, padrao2, saida0)
    print(neuronio, "saida0 =", saida_0)
    neuronio, saida_0 = ajustes(neuronio, padrao3, saida0)
    print(neuronio, "saida0 =", saida_0)
    neuronio, saida_0 = ajustes(neuronio, padrao4, saida0)
    print(neuronio, "saida0 =", saida_0)
    neuronio, saida_0 = ajustes(neuronio, padrao5, saida0)
    print(neuronio, "saida0 =", saida_0)
    
    # Treina com jogadores Vencedores (Classe 1)
    neuronio, saida_1 = ajustes(neuronio, padrao6, saida1)
    print(neuronio, "saida1 =", saida_1)
    neuronio, saida_1 = ajustes(neuronio, padrao7, saida1)
    print(neuronio, "saida1 =", saida_1)
    neuronio, saida_1 = ajustes(neuronio, padrao8, saida1)
    print(neuronio, "saida1 =", saida_1)
    neuronio, saida_1 = ajustes(neuronio, padrao9, saida1)
    print(neuronio, "saida1 =", saida_1)
    neuronio, saida_1 = ajustes(neuronio, padrao10, saida1)
    print(neuronio, "saida1 =", saida_1)
    neuronio, saida_1 = ajustes(neuronio, padrao11, saida1)
    print(neuronio, "saida1 =", saida_1)
    
    n += 1
    print("Número de ciclos =", n)

# Extração dos pontos para plotagem
# Classe 0: jogadores Perdedores (padrao0 a padrao5)
# Classe 1: jogadores Vencedores (padrao6 a padrao11)
poder_class0 = [padrao0[1], padrao1[1], padrao2[1], padrao3[1], padrao4[1], padrao5[1]]
agilidade_class0 = [padrao0[2], padrao1[2], padrao2[2], padrao3[2], padrao4[2], padrao5[2]]
poder_class1 = [padrao6[1], padrao7[1], padrao8[1], padrao9[1], padrao10[1], padrao11[1]]
agilidade_class1 = [padrao6[2], padrao7[2], padrao8[2], padrao9[2], padrao10[2], padrao11[2]]

# Geração do gráfico que ilustra a classificação dos jogadores
plt.figure(figsize=(8, 6))
plt.scatter(poder_class0, agilidade_class0, color='blue', label='Perdedores')
plt.scatter(poder_class1, agilidade_class1, color='red', label='Vencedores')

# Cálculo da fronteira de decisão:
# Equação: -w0 + w1*Poder + w2*Agilidade = 0  =>  Agilidade = (w0 - w1*Poder) / w2
w0, w1, w2 = neuronio
poder_range = np.linspace(min(poder_class0), max(poder_class1), 100)
agilidade_boundary = (w0 - w1 * poder_range) / w2
plt.plot(poder_range, agilidade_boundary, color='green', linestyle='--', label='Fronteira de Decisão')

plt.xlabel("Poder")
plt.ylabel("Agilidade")
plt.title("Classificação de Jogadores - Marvel Rivals")
plt.legend()
plt.grid(True)
plt.show()

# Teste de Generalização com novos jogadores (novos desempenhos simulados)
p0 = [-1, 25, 17]   # Jogador com performance baixa (esperado Perdedor)
p1 = [-1, 30, 20]   # Jogador com performance baixa (esperado Perdedor)
p2 = [-1, 70, 27]   # Jogador com performance intermediária (pode ser Vencedor)
p3 = [-1, 95, 34]   # Jogador com performance alta (esperado Vencedor)
p4 = [-1, 50, 22]   # Jogador com desempenho intermediário

print("Testes de Generalização:")
neuronio, saida_0 = teste_generalizacao(neuronio, p0, saida0)
print(neuronio, "saida =", saida_0, "->", "Perdedor" if saida_0 == 0 else "Vencedor")
neuronio, saida_0 = teste_generalizacao(neuronio, p1, saida0)
print(neuronio, "saida =", saida_0, "->", "Perdedor" if saida_0 == 0 else "Vencedor")
neuronio, saida_0 = teste_generalizacao(neuronio, p2, saida0)
print(neuronio, "saida =", saida_0, "->", "Perdedor" if saida_0 == 0 else "Vencedor")
neuronio, saida_0 = teste_generalizacao(neuronio, p3, saida0)
print(neuronio, "saida =", saida_0, "->", "Perdedor" if saida_0 == 0 else "Vencedor")
neuronio, saida_0 = teste_generalizacao(neuronio, p4, saida0)
print(neuronio, "saida =", saida_0, "->", "Perdedor" if saida_0 == 0 else "Vencedor")
