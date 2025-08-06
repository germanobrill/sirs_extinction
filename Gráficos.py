import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

def plotar_simulacao(grupo_simulacao, nome_arquivo_saida=None):
    """
    Plota as séries temporais S, I, R, V de um único grupo de simulação HDF5.

    Argumentos:
        grupo_simulacao (h5py.Group): O grupo do arquivo HDF5 contendo os dados da simulação.
        nome_arquivo_saida (str, opcional): Se fornecido, salva o gráfico em um arquivo com este nome.
    """
    # 1. Extrai os dados dos datasets
    tempo = grupo_simulacao['tempo'][:]
    S = grupo_simulacao['S'][:]
    I = grupo_simulacao['I'][:]
    R = grupo_simulacao['R'][:]
    V = grupo_simulacao['V'][:]

    # 2. Extrai os parâmetros dos atributos para o título do gráfico
    params = grupo_simulacao.attrs
    r0 = params.get('r0', 'N/A')
    delta = params.get('delta', 'N/A')
    a = params.get('a', 'N/A')
    t_final = params.get('t_final', tempo[-1])
    motivo_parada = params.get('motivo_parada', 'N/A')

    # 3. Cria o gráfico
    plt.style.use('seaborn-v0_8-whitegrid') # Estilo visual do gráfico
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(tempo, S, label='Suscetíveis (S)', color='blue')
    ax.plot(tempo, I, label='Infectados (I)', color='red')
    ax.plot(tempo, R, label='Recuperados (R)', color='green')
    ax.plot(tempo, V, label='Vacinados Acum. (V)', color='purple', linestyle='--')

    # 4. Configura os detalhes do gráfico
    ax.set_title(f'Modelo SIRS ($R_0={r0:.1f}, \\delta={delta:.4f}, \\alpha={a:.3f}$)\nMotivo da Parada: {motivo_parada}', fontsize=16)
    ax.set_xlabel('Tempo', fontsize=12)
    ax.set_ylabel('Fração da População', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, t_final) # Limita o eixo x ao tempo final da simulação
    ax.set_ylim(0, 1)   # Limita o eixo y entre 0 e 1

    # 5. Mostra ou salva o gráfico
    if nome_arquivo_saida:
        plt.savefig(nome_arquivo_saida, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em '{nome_arquivo_saida}'")
        plt.close(fig) # Fecha a figura para não exibi-la na tela
    else:
        plt.show()

# =============================================================================
# BLOCO PRINCIPAL DE EXECUÇÃO
# =============================================================================

# Verifique se o nome do arquivo e o caminho estão corretos
# Tenta encontrar o arquivo na Área de Trabalho (Desktop)
try:
    home_dir = os.path.expanduser('~')
    # Se seu sistema estiver em inglês, troque 'Área de Trabalho' por 'Desktop'
    caminho_do_arquivo = os.path.join('Simulações completas.h5')
    arquivo_h5 = h5py.File(caminho_do_arquivo, 'r')
except FileNotFoundError:
    print(f"ERRO: Arquivo '{caminho_do_arquivo}' não encontrado.")
    print("Verifique se o nome e o caminho para o arquivo HDF5 estão corretos.")
    exit()


# --- ESCOLHA UMA DAS OPÇÕES ABAIXO ---

# --- OPÇÃO A: Plotar uma única simulação específica ---
print("Plotando uma única simulação (sim_0)...")
simulacoes_disponiveis = list(arquivo_h5.keys())
if 'sim_2' in simulacoes_disponiveis:
    grupo_especifico = arquivo_h5['sim_1']
    plotar_simulacao(grupo_especifico)
else:
    print("Erro: 'sim_0' não encontrado no arquivo. Simulações disponíveis:", simulacoes_disponiveis)


# --- OPÇÃO B: Plotar TODAS as simulações e salvar em arquivos separados ---
# Descomente as linhas abaixo para ativar esta opção.

# print("\nPlotando todas as simulações e salvando em arquivos...")
# output_dir = "graficos_simulacoes"
# os.makedirs(output_dir, exist_ok=True) # Cria uma pasta para os gráficos

# for nome_sim in arquivo_h5.keys():
#     grupo = arquivo_h5[nome_sim]
#     nome_arquivo = os.path.join(output_dir, f"{nome_sim}.png")
#     plotar_simulacao(grupo, nome_arquivo_saida=nome_arquivo)

# -----------------------------------------

# É importante fechar o arquivo HDF5 após o uso
arquivo_h5.close()