import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_ivp
import pandas as pd
from numba import njit
from joblib import Parallel, delayed 
import itertools
from tqdm import tqdm

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
    t_total = 1250
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
        mplt_step=0.1
    )
    t_eval=np.linspace(delta_t[0], delta_t[1], int(t_total / dt))
    i = sol.y[1]
    s = sol.y[0]
    min_infected = min(i)
    if min_infected < 1/n:
        prob_ext = 1 - min_infected
    else:
        prob_ext = 0.0
    return [s,i,t_eval, tf, prob_ext]

N = 1e4
R0 = [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9] # Valores de R0
vac_num = 0.7 # Número de vacinas
GAMMA = 0.1 # Taxa de recuperação
DELTA = 0.005 # Taxa de perda de imunidade
t0 = 160 # Data de início da vacinação
a = 0.01 #Taxa de vacinação


param_combinations = []
for r0 in R0:
    params = (N, r0, GAMMA, DELTA, vac_num, t0, a)
    param_combinations.append(params)

def arredondar_significativos(x, alg=4):
    if isinstance(x, (int, float, np.number)):
        if x == 0:
            return 0
        else:
            return round(x, -int(np.floor(np.log10(abs(x)))) + (alg - 1))
    return x  # retorna valor original se não for número

resultados = Parallel(n_jobs=-1)(
        delayed(simular_uma_vez)(*params) for params in tqdm(param_combinations))        

Ixt_pdf = 'Ixt para N=1e4 (t0 = 160).pdf'

with PdfPages(Ixt_pdf) as pdf:
    for i in range(len(R0)):
        # Acessar o resultado correto (i) e extrair os dados
        resultado_atual = resultados[i]
        susceptibles = resultado_atual[0]
        infected = resultado_atual[1]
        time = resultado_atual[2]
        TF = resultado_atual[3]
        prob_ext = arredondar_significativos(resultado_atual[4], alg=4)
        
        r0_atual = R0[i]
        plt.plot(time, infected)
        plt.axvspan(t0, TF, color='lightblue', alpha=0.7, label="Vaccination Period")
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Infected Ratio')
        plt.grid(True)
        plt.title(f'Infected ratio for R= {r0_atual} (N = 1e4)', fontsize=18)
        params_legend_text = (
        f'Parameters:\n'
        f'vac_num = {vac_num}\n'
        f'$\\gamma$ = {GAMMA}\n'
        f'$\\delta$ = {DELTA}\n'
        f'$t_0$ = {t0}\n'
        f'$\\alpha$ = {a}\n'
        f'Extinction Probability = {prob_ext}'
        )
        plt.legend(title=params_legend_text, loc='upper right', fontsize=10)
        pdf.savefig()
        plt.close()
        print(f"Plot for R0 = {r0_atual} saved to PDF.")
    print(f"All plots saved to {Ixt_pdf}.")


