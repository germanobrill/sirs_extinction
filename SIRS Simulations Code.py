
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from numba import njit
from joblib import Parallel, delayed
import itertools

t_total = 750 # Simulation duration
dt = 0.01 # Step size
t_values = np.arange(0, t_total, dt) # Array containing the time values

@njit
def alpha(t0, tf, t, a):
    if t0 <= t <= tf:
        return a
    else:
        return 0.0

def SIRS_numba(t, y, beta, mu, delta, t0, tf, a):
    S,I,R = y
    vac = alpha(t0, tf, t, a)
    dSdt = -beta*S*I + delta*R - vac
    dIdt = beta*S*I -mu*I
    dRdt = mu*I - delta*R + vac
    return np.array([dSdt, dIdt, dRdt])

def SIRS_wrapper(t, y, beta, mu, delta, t0, tf, a):
    return SIRS_numba(t, y, beta, mu, delta, t0, tf, a)

def simular_uma_vez(n, r0, mu, delta, vac_num, t0, a):
    t_total = 750
    dt = 0.1
    beta = mu * r0
    i_0 = 1 / n
    s_0 = 1 - i_0
    r_0 = 0
    time_window = vac_num / a
    tf = t0 + time_window
    delta_t = (0, t_total)

    sol = solve_ivp(
        SIRS_wrapper,
        delta_t,
        [s_0, i_0, r_0],
        method='RK45',
        t_eval=np.linspace(delta_t[0], delta_t[1], int(t_total / dt)),
        args=(beta, mu, delta, t0, tf, a),
        atol=1e-10,
        rtol=1e-6,
        max_step=0.1
    )

    return [n, r0, mu, delta, vac_num, t0, a, min(sol.y[1])]

N = np.array([1e4, 1e5, 1e6, 1e7])
R0 = np.array([1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 5, 7, 9, 11 , 13 , 15]) # Número de reprodução básico
GAMMA = np.array([0.05, 0.1, 0.15, 0.2]) # Taxa de cura
DELTA = np.array([0.006, 0.008]) # Taxa de perda de imunidade
VAC_NUM = np.array([0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5]) # Número de vacinas disponíveis (Normalizado)
T0 = np.array([20, 40, 60, 80, 100, 120, 140]) # Data de início da vacinação
A = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]) #Taxa de vacinação

from tqdm import tqdm

param_combinations = list(itertools.product(N, R0, GAMMA, DELTA, VAC_NUM, T0, A))
resultados = Parallel(n_jobs=-1)(
    delayed(simular_uma_vez)(*params) for params in tqdm(param_combinations)
)
df = pd.DataFrame(resultados)
df.to_csv('Minimo de infectados delta 0,006 e 0,008.csv', index=False, header=False)