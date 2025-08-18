import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
from numba import njit
from joblib import Parallel, delayed
import itertools
import h5py

t_total = 2000.0 # Duração máxima das simulações

'''A função calcular_taxa_vac decide qual a taxa de vacinação em cada instante, com base nas condições do sistema.'''

@njit
def calcular_taxa_vac(t, S, V, t0, a, vac_num):
    # Condição de parada: o número de vacinados (V) atingiu o estoque total?
    if t > t0 and V < vac_num:
        return a  # Retorna a taxa de vacinação
    else:
        return 0.0 # Sem vacinação (taxa zero)
    
@njit
def SIRS_numba(t, y, beta, mu, delta, t0, vac_num, a):
    S,I,R,V = y
    taxa_vac = calcular_taxa_vac(t, S, V, t0, a, vac_num)
    dSdt = -beta*S*I + delta*R - taxa_vac
    dIdt = beta*S*I -mu*I
    dRdt = mu*I - delta*R + taxa_vac
    dVdt = taxa_vac
    return np.array([dSdt, dIdt, dRdt, dVdt])

''' Abaixo definimos um evento que vai indicar uma condição de parada na simulação.
Quando a norma euclidiana das derivadas se aproxima de um valor muito pequeno (threshold)
a função retorna um valor negativo, indicando a parada da simulação. '''

@njit
def evento_estado_estacionario(t, y, beta, mu, delta, t0, vac_num, a):
    S,I,R,V = y
    taxa_vac = calcular_taxa_vac(t, S, V, t0, a, vac_num)
    dSdt = -beta * S * I + delta * R - taxa_vac
    dIdt = beta * S * I - mu * I
    dRdt = mu * I - delta * R + taxa_vac
    dVdt = taxa_vac
    threshold = 1e-7
    norma_derivadas = np.sqrt(dSdt**2 + dIdt**2 + dRdt**2 + dVdt**2)
    
    return norma_derivadas - threshold

''' Abaixo definimos um outro evento que vai indicar uma condição de parada na simulação.
Quando a S se torna menor que 0.005, a simulação é interrompida. '''

@njit
def evento_parada_S_minimo(t, y, beta, mu, delta, t0, vac_num, a):
    S = y[0]  # Acessamos S, que é o primeiro elemento do vetor y
    return S - 0.005

'''A função abaixo realiza a simulação do modelo SIRS usando o solve_ivp com base na função SIRS_numba,
parando assim que a função evento_estado_estacionario retornar um valor negativo. Após concluída a
simulação, a função retorna toda a série temporal de S, I, R e V, além de retornar os parâmetros usados.'''

def simular_uma_vez(n, r0, mu, delta, vac_num, t0, a):
    dt = 0.1
    beta = mu * r0
    i_0 = 1 / n
    s_0 = 1 - i_0
    r_0 = 0.0
    v_0 = 0.0
    delta_t = (0, t_total)

    evento_estado_estacionario.terminal = True
    evento_estado_estacionario.direction = -1
    evento_parada_S_minimo.terminal = True
    evento_parada_S_minimo.direction = -1

    sol = solve_ivp(
        SIRS_numba,
        delta_t,
        [s_0, i_0, r_0, v_0],
        method='RK45',
        t_eval=np.linspace(delta_t[0], delta_t[1], int(t_total / dt)),
        args=(beta, mu, delta, t0, vac_num, a),
        events= [evento_estado_estacionario,evento_parada_S_minimo],
        atol=1e-10,
        rtol=1e-6,
        max_step=0.1
    )
    # Definindo a probabilidade de Extinção:
    prob_ext = np.where(min(sol.y[1]) < 1/n, 1 - min(sol.y[1])*n, 0)
    params_dict = {'n': n, 'r0': r0, 'mu': mu, 'delta': delta, 'vac_num': vac_num, 't0': t0, 'a': a, 
                   'prob_ext' : prob_ext, 't_final' : sol.t[-1], 't_pico' : sol.t[np.argmax(sol.y[1])],
                   'motivo_parada' : 'Estado Estacionário' if sol.t_events[0].size > 0 else ('S Mínimo' if sol.t_events[1].size > 0 else 't_total')}
    resultado = {
            'params': {'n': n, 'r0': r0, 'mu': mu, 'delta': delta, 'vac_num': vac_num, 't0': t0, 'a': a},
            'resultados': {'prob_ext': prob_ext, 't_final': sol.t[-1], 't_pico': sol.t[np.argmax(sol.y[1])], 
                    'motivo_parada': 'Estado Estacionário' if sol.t_events[0].size > 0 else ('S Mínimo' if sol.t_events[1].size > 0 else 't_total')},
            'tempo': sol.t,
            'S': sol.y[0],
            'I': sol.y[1],
            'R': sol.y[2],
            'V': sol.y[3]
        }
    return resultado

'''Lista de todos os parâmetros que se deseja realizar simulações'''

N = np.array([1e4]) # Tamanho da população 
R0 =  np.arange(1.1, 3, 0.02)# Número de reprodução básico
GAMMA = np.array([0.1]) # Taxa de cura
DELTA = np.array([0.005]) # Taxa de perda de imunidade
VAC_NUM = np.arange(0.2, 1.5, 0.02) # Número de vacinas disponíveis (Normalizado)
T0 = np.array([80.0]) # Data de início da vacinação
A = np.array([0.01]) #Taxa de vacinação

'''O tqdm será usado para indicar quantas das simulações já foram realizadas e quantas iterações são realizadas 
por segundo, o que permite um bom controle da eficiência e funcionamento do código.'''

from tqdm import tqdm

'''Abaixo é realizada a simulação para todas as combinação de parâmetros:'''

param_combinations = list(itertools.product(N, R0, GAMMA, DELTA, VAC_NUM, T0, A))
resultados_completos = Parallel(n_jobs=-1)(
    delayed(simular_uma_vez)(*params) for params in tqdm(param_combinations)
)

''' Aqui usaremos o h5py, que permite salvar dados de formatos distintos de maneira bastante
organizada e de fácil manipulação em python.''' 

with h5py.File('Simulações mapa de calor.h5', 'w') as f:
    for i, res in enumerate(resultados_completos):
        if res is None:
            continue

        grp = f.create_group(f'sim_{i}')
        
        for key, value in res['params'].items():
            grp.attrs[key] = value

        for key, value in res['resultados'].items():
            grp.attrs[key] = value
            
        # Acessamos diretamente as chaves do dicionário que criamos
        grp.create_dataset('tempo', data=res['tempo'])
        grp.create_dataset('S', data=res['S'])
        grp.create_dataset('I', data=res['I'])
        grp.create_dataset('R', data=res['R'])
        grp.create_dataset('V', data=res['V'])