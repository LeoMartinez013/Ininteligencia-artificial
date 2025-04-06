```python
# ============================================================================
# Simulação da Partida: Cálculo da razão eliminações/mortes para determinar
# quem foi o melhor (MVP) e o pior da partida.
# Os dados simulados serão gerados em valores normalizados [0,1]
# ============================================================================
placar = []
for i in range(12):
    # Para simulação, eliminações variam de 0 até 1 (simulando um máximo de 10 eliminações)
    eliminacoes = random.uniform(0, 1)
    # Mortes variam de 0.1 até 1 (garantindo que não seja zero)
    mortes = random.uniform(0.1, 1)
    razao = eliminacoes / mortes
    placar.append((f"Jogador {i+1}", eliminacoes, mortes, razao))
# Ordena os jogadores pela razão (maior razão indica melhor desempenho)
placar.sort(key=lambda x: x[3], reverse=True)
print("\nPlacar Final da Partida (baseado na razão Eliminações/Mortes):")
for pos, (nome, elim, mort, razao) in enumerate(placar, start=1):
    print(f"{pos}: {nome} - Eliminacoes: {elim:.2f}, Mortes: {mort:.2f}, Razão: {razao:.2f}")
print("\nMelhor jogador (MVP):", placar[0][0])
print("Pior jogador:", placar[-1][0])
```