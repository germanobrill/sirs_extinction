# Importing libraries
import numpy as np
import pandas as pd
import numba as nb
import h5py

# Defining class parameters type because numba is too fussy
spec = [('n_agents', nb.i8), ('S_i', nb.i8), ('I_i', nb.i8),
        ('beta', nb.f8), ('gamma', nb.f8), ('delta', nb.f8),
        ('t_start', nb.f8), ('t_stop', nb.f8), ('alpha', nb.f8), ('h', nb.f8),
        ('S_serie', nb.types.ListType(nb.types.int64)),
        ('I_serie', nb.types.ListType(nb.types.int64)),
        ('R_serie', nb.types.ListType(nb.types.int64)),
        ('t_serie', nb.types.ListType(nb.types.float64)),
        ('agents', nb.i8[:])]

# Creating class using jitclass, which as at experimental state in feb 2025
@nb.experimental.jitclass(spec)
class model_agents_numba():

  def __init__(self, n_agents, S_i, I_i, beta, gamma, delta, t_start, t_stop, alpha, h):
    # Create object properties
    self.n_agents = n_agents
    self.beta = beta
    self.gamma = gamma
    self.delta = delta
    self.t_start = t_start
    self.t_stop = t_stop
    self.alpha = alpha
    self.h = h

    # Create list with groups values
    R_i = n_agents - S_i - I_i
    self.S_serie = nb.typed.List([S_i])
    self.I_serie = nb.typed.List([I_i])
    self.R_serie = nb.typed.List([R_i])
    self.t_serie = nb.typed.List([0.0])

    # Create agents. state = 0 -> S. state = 1 -> I. state = 2 -> R
    state = np.array([0]*S_i + [1]*I_i + [2]*R_i)
    self.agents = state

  def evolve(self):

    # Get index of agents in different groups
    agents_temp = self.agents

    sus_index = np.where(agents_temp == 0)[0]
    inf_index = np.where(agents_temp == 1)[0]
    rem_index = np.where(agents_temp == 2)[0]

    # Change suseptible into infected and removed
    inf_probability = self.beta * len(inf_index)/self.n_agents * self.h
    if self.t_serie[-1] > self.t_start and self.t_serie[-1] < self.t_stop:
      rem_probability = self.alpha * self.n_agents/len(sus_index) * self.h
    else:
      rem_probability = 0
    x = np.random.uniform(0,1,len(sus_index))
    cond_list = [x <= inf_probability, x <= inf_probability + rem_probability]
    suscepted = np.where(cond_list[0], 1, np.where(cond_list[1], 2, 0))
    agents_temp[sus_index] = suscepted

    # Change infected into removed
    rem_probability = self.gamma * self.h
    change_condition = np.random.uniform(0, 1, len(inf_index)) <= rem_probability
    infected = np.where(change_condition, 2, 1)
    agents_temp[inf_index] = infected

    # Change removed into suceptible
    sus_probability = self.delta * self.h
    change_condition = np.random.uniform(0, 1, len(rem_index)) <= sus_probability
    removed = np.where(change_condition, 0, 2)
    agents_temp[rem_index] = removed

    self.agents = agents_temp

  # Function to evolve the system a defined number of steps, saving time series
  # of S, I, R and t in the process
  def step(self, steps):

    for _ in range(steps):

      self.evolve()

      states = self.agents

      S = np.sum(states == 0)
      I = np.sum(states == 1)
      R = np.sum(states == 2)

      self.S_serie.append(S)
      self.I_serie.append(I)
      self.R_serie.append(R)
      self.t_serie.append(self.t_serie[-1]+self.h)

  # Get series of t, S, I and R all at once
  def get_series(self):

    tt = np.asarray(self.t_serie)
    SS = np.asarray(self.S_serie)/self.n_agents
    II = np.asarray(self.I_serie)/self.n_agents
    RR = np.asarray(self.R_serie)/self.n_agents

    return tt, SS, II, RR

'''
 Function to run multiple simulations with agents according to the "param" array,
 which must be an NxM array.
 By the way, M = 7, because the columns are organize as follows
     ([beta, R_0, delta, t_start, t_stop, alpha, repetition_index], ...)
 where repetition_index are are useful to differenciate simulations using the
 same parameter values.
'''
@nb.jit('Tuple((f8[:,:], f8[:], f8[:], f8[:], f8[:], f8[:]))'
        '(i8, i8, i8, f8[:,:], f8, f8)',
        nopython = True, parallel = True)
def func_sim_agents_numba(N, S_i, I_i, params, h, t_end):

  # Number of rows in "params" = number of simulations
  N_sims = len(params)
  N_steps = int(t_end/h + 1)

  # Create two-dimentional arrays to save multiple time series

  I_min_list = np.empty(N_sims, dtype=np.float64)
  t_pico_list = np.empty(N_sims, dtype=np.float64)
  gamma_list = np.empty(N_sims, dtype=np.float64)
  vac_num_list = np.empty(N_sims, dtype=np.float64)
  S_min_list = np.empty(N_sims, dtype=np.float64)
  
  for i in nb.prange(N_sims):

      # Get current parameters to use and calculate gamma and the vaccines number
      b, r_0, d, t_sa, t_so, a, n = params[i]
      g = b/r_0
      v = (t_so - t_sa)/a

      # Simulate
      epi_agents_v = model_agents_numba(N, S_i, I_i, b, g, d, t_sa, t_so, a, h)

      epi_agents_v.step(int(t_end/h))

      # Colect simulation 1D arrays and add to 2D arays
      tt, S, I, R = epi_agents_v.get_series()
      S_min_list[i] = np.min(S)
      I_min_list[i] = np.min(I)
      t_pico_list[i] = tt[np.argmax(I)]
      gamma_list[i] = g
      vac_num_list[i] = v
  return params, I_min_list, t_pico_list, gamma_list, vac_num_list, S_min_list

# Function to save simulation data. "file_format" must be 'npz' (compressed), 'csv' (no compressed) or 'h5' (python dictionary)
def save_sim(sim, name, file_format):
    try:
        params, I_min_list, t_pico_list, gamma_list, vac_num_list, S_min_list = sim
    except ValueError as e:
        print(f"Erro ao desempacotar dados da simulação. Verifique o retorno de 'func_sim_agents_numba'. {e}")
        return

    match file_format:
        case 'npz':
            # Salva os parâmetros e os arrays de resumo
            np.savez_compressed(name,
                    parameters = params,
                    I_min_list = I_min_list,
                    t_pico_list = t_pico_list,
                    gamma_list = gamma_list,
                    vac_num_list = vac_num_list,
                    S_min_list = S_min_list)

        case 'csv':
            # Combina os parâmetros e os resumos em um DataFrame
            df_params = pd.DataFrame(params, columns=['beta', 'r_0', 'delta', 't_start', 't_stop', 'alpha_cte', 'n_sample'])
            df_resumo = pd.DataFrame({
                'I_min': I_min_list,
                't_pico': t_pico_list,
                'gamma': gamma_list,
                'vac_num': vac_num_list,
                'S_min': S_min_list
            })
            df = pd.concat([df_params, df_resumo], axis=1)
            df.to_csv(f'{name}.csv', index=False)
        
        case 'h5':
            # Itera sobre os resumos, salvando um grupo por simulação
            sim_data_iterator = zip(
                params, I_min_list, t_pico_list, gamma_list, vac_num_list, S_min_list
            )
            with h5py.File(f'{name}.h5', 'w') as f:
                for i, (p_row, i_min_val, t_pico_val, g_val, v_val, s_min_val) in enumerate(sim_data_iterator):
                    
                    # Cria um grupo para cada simulação
                    grp = f.create_group(f'sim_{i}')
                    
                    # Salva parâmetros de entrada como atributos do grupo
                    grp.attrs['beta'] = p_row[0]
                    grp.attrs['r_0'] = p_row[1]
                    grp.attrs['delta'] = p_row[2]
                    grp.attrs['t_start'] = p_row[3]
                    grp.attrs['t_stop'] = p_row[4]
                    grp.attrs['alpha'] = p_row[5]
                    grp.attrs['n'] = p_row[6]
                    
                    # Salva parâmetros calculados como atributos
                    grp.attrs['gamma'] = g_val
                    grp.attrs['vac_num'] = v_val

                    # Salva resultados resumidos como atributos
                    grp.attrs['I_min'] = i_min_val
                    grp.attrs['t_pico'] = t_pico_val
                    grp.attrs['S_min'] = s_min_val
'''
 Run simulations
'''

# FIST SIMULATION JUST TO START NUMBA MODEL
S_i, I_i, R_i = 9999, 1, 0
N = S_i + I_i + R_i

beta, R_0, delta = np.array([0.1]), np.array([0.5]), np.array([0.01])

t_start, t_stop, alpha = np.array([400.0]), np.array([600.0]), np.array([0.003])

sample_size = 1

t_end = 1250

h = 1e-0

params = np.array([[b, r_0, d, t_sa, t_so, a, n]
                   for b in beta for r_0 in R_0 for d in delta for t_sa in t_start for t_so in t_stop for a in alpha for n in range(sample_size)],
                    dtype = np.float64)

func_sim_agents_numba(N, S_i, I_i, params, h, t_end)


# RUN AND SAVE THE FOLLOWING SIMULATIONS
S_i, I_i, R_i = 9990, 10, 0
N = S_i + I_i + R_i

gamma = 0.1
R_0, delta = np.arange(1.1, 3.0, 0.01), np.array([0.005])
beta = gamma*R_0

t_start, t_stop, alpha = np.array([80.0]), np.arange(100.0, 230.0, 1.0), np.array([0.01])

sample_size = 25 # Number of repetitions with ne same parameter values

t_end = 1250

h = 5*1e-1

params = np.array([[b, r_0, d, t_sa, t_so, a, n]
                   # Correção: zip(beta, R_0)
                   for (b, r_0) in zip(beta, R_0) 
                   for d in delta for t_sa in t_start for t_so in t_stop for a in alpha for n in range(sample_size)],
                    dtype = np.float64)

sim = func_sim_agents_numba(N, S_i, I_i, params, h, t_end)

# Save localy. If you already have a file with the same name it will be overwriten
name, file_format = 'Heatmap_MBA_1e4', 'h5'
save_sim(sim, name, file_format)

del sim

