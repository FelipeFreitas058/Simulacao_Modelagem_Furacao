import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import autograd.numpy as anp
from autograd import jacobian
from scipy.ndimage import uniform_filter1d

########### Definição de parâmetros do script ###################################################################################################################################################################################################################

Raio_maximo = 1000              # Valor máximo do raio da curva C à ser calculado em km
Precisao = 100                  # Numero de pontos no qual 'R' será dividido. É diretamente proporcional à precisão do gráfico a ser gerado

########### Definição de parâmetros do furacão ##################################################################################################################################################################################################################

v_mc_x = 0                      # Velocidade de deslocamento do centro do furacão em x
v_mc_y = 7                      # Velocidade de deslocamento do centro do furacão em x

P_n = 1013.25                   # Pressão ambiente
P_c = 840                       # Pressão no centro

L = 24.8                        # Latitude do centro do furacão

R_max = -1                      # raio do vortice (km)          {-1 para calcular o valor através de um modelo}
B_h = -1                        # parâmetro de Holland          {-1 para calcular o valor através de um modelo}



Saida = "Circulação"            # Característica a ser calculada (opções: "Fluxo" ou "Circulação")

#################################################################################################################################################################################################################################################################

def V(X, v_mc_x, v_mc_y, P_n, P_c, R_max, B_h, L):                          # Função que calcula e retorna as componentes x e y da velocidade total do furacão
    x, y = X

    r = np.sqrt(x**2 + y**2)*1000
    Theta = np.arctan2(y,x)
    P_n = P_n*100
    P_c = P_c*100
    r = np.maximum(r, 1e-6)

    v_mc = np.sqrt(v_mc_x**2 + v_mc_y**2)
    V_mov_x = v_mc_x*np.exp(-r/500000)
    V_mov_y = v_mc_y*np.exp(-r/500000)
    D_p = P_n - P_c

    if R_max == -1:
        R_max = (56.92 -0.1541*v_mc + 0.7372*(L-25))
    R_max = R_max*1000

    if B_h == -1:
        B_h = 1.5 + ((98000 - P_c) / 12000)
        
    A_h = R_max**(B_h)
    f_c = 2*(7.2921*10**(-5)*np.sin(L*np.pi/180))
    pho = 1.15

    exp_term = np.exp(-A_h / r**B_h)
    numerador = A_h * B_h * D_p * exp_term
    denominador = pho * r**B_h
    V_r_quadrado = (numerador / denominador) + (r**2 * f_c**2) / 4
    V_r = np.sqrt(V_r_quadrado) - (r * f_c) / 2
    V_r = np.nan_to_num(V_r)

    V_rot_x = -V_r * np.sin(Theta)
    V_rot_y = V_r * np.cos(Theta)

    V_x = V_rot_x + V_mov_x
    V_y = V_rot_y + V_mov_y
    return np.array([V_x, V_y])

def C(t,R):                                                                 # Função que retorna as funções componentes da parametrização da curva C sobre a qual será integrada                           
    return anp.array([R*anp.cos(t), R*anp.sin(t)])              

dC_dt = jacobian(C,0)                                                       # Definição do vetor tangente, não unitário, T(t) à curva C

def Magnitude(v):                                                           # Função que retorna a magnitude de um vetor v passado por referência
    return (anp.sqrt(v[0]**2 + v[1]**2))

def Vetor_Normal_Tangente(t,R):                                             # Função que retorna o vetor tangente, e unitário, à curva C caso 'Saida = "Circulação"', e retorna o vetor normal, e unitário, à curva C caso 'Saida = "Fluxo"'
    dx, dy = dC_dt(t,R)
    vetor = Magnitude(dC_dt(t,R))
    if Saida == "Circulação":
        return anp.array([dx / vetor, dy / vetor])
    elif Saida == "Fluxo":
        return anp.array([-dy / vetor, dx / vetor])


def Integrando(t,R, v_mc_x, v_mc_y, P_n, P_c, R_max, B_h, L):               # Função que calcula e retorna o valor do integrando
    V_Total = V(C(t,R), v_mc_x, v_mc_y, P_n, P_c, R_max, B_h, L)
    Normal_Tangente = Vetor_Normal_Tangente(t,R)
    magnitude = Magnitude(dC_dt(t,R))
    return (anp.dot(V_Total, Normal_Tangente) * magnitude)


resultados_F_C = []
valores_R = []
Valores_t = np.linspace(0, 2*np.pi, 100)

for R in np.linspace(1, Raio_maximo, Precisao):                                         # Laço 'for' que percorre diferentes valores de 'R' (respectivo raio da curva C), para que o fluxo/circulação seja calculado em todo o furacão.
    Valores_Integrando = [] 
    for t in Valores_t:
        Valores_Integrando.append(Integrando(t, R, v_mc_x, v_mc_y, P_n, P_c, R_max, B_h, L))
    Fluxo_Circulacao = simpson(Valores_Integrando, x=Valores_t)
    resultados_F_C.append(Fluxo_Circulacao)
    valores_R.append(R)

plt.figure(figsize=(10, 10))
plt.plot(valores_R, resultados_F_C, linestyle='-', color='b')
plt.show()
