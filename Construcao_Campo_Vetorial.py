import numpy as np
import matplotlib.pyplot as plt

Intervalos = 300                # Limite do eixo X e Y do gráfico em km
Densidade_Vetores = 50
Escala_Grafico = 1000            # Escala do gráfico

########### Definição de parâmetros do furacão ###########################################################################

v_mc_x = 0                      # Velocidade de deslocamento do centro do furacão em x
v_mc_y = 21                      # Velocidade de deslocamento do centro do furacão em x

P_n = 1013.25                   # Pressão ambiente
P_c = 840                       # Pressão no centro

L = 24.8                        # Latitude do centro do furacão

R_max = -1                      # raio do vortice (km)          {-1 para calcular o valor através de um modelo}
B_h = -1                        # parâmetro de Holland          {-1 para calcular o valor através de um modelo}

##########################################################################################################################

P_c = P_c*100                                                       # Conversão de hPa para Pa
P_n = P_n*100                                                       # Conversão de hPa para Pa

x = np.linspace(-Intervalos, Intervalos, Densidade_Vetores)         # Conjunto de pontos para a contrução do gráfico
y = np.linspace(-Intervalos, Intervalos, Densidade_Vetores)         # Conjunto de pontos para a contrução do gráfico
X, Y = np.meshgrid(x, y)

r = np.sqrt(X**2 + Y**2)*1000                                       # Cálculo de 'r' através da conversão de coordenadas retangulares para polares.
r[r == 0] = 1e-10
Theta = np.arctan2(Y, X)                                            # Cálculo de 'Theta' através da conversão de coordenadas retangulares para polares.

v_mc = np.sqrt(v_mc_x**2 + v_mc_y**2)                               # Cálculo do módulo do vetor de velocidade de deslocamento do centro do furacão

V_mov_x = v_mc_x*np.exp(-r/500000)                                  # Cálculo da coordenada x da velocidade de deslocamento do furacão
V_mov_y = v_mc_y*np.exp(-r/500000)                                  # Cálculo da coordenada y da velocidade de deslocamento do furacão

D_p = P_n - P_c                                                     # Cálculo da diferença de pressão entre o centro do furacão e o exterior

if R_max == -1:
    R_max = (56.92 -0.1541*v_mc + 0.7372*(L-25))                    # Cálculo do raio do vórtice, caso 'R_max' seja igual à -1
R_max = R_max*1000                                                  # Conversão de km para m

if B_h == -1:
    B_h = 1.5 + ((98000 - P_c) / 12000)                             # Cálculo do parâmetro B de Holland, caso 'B_h' seja igual à -1
    
A_h = R_max**(B_h)
f_c = 2*(7.2921*10**(-5)*np.sin(L*np.pi/180))                       # Cálculo da força de Coriolis para a latitude arbitrada
pho = 1.15

exp_term = np.exp(-A_h / r**B_h)                                    # Aplicação do modelo de Holland
numerator = A_h * B_h * D_p * exp_term                              # Aplicação do modelo de Holland
denominator = pho * r**B_h                                          # Aplicação do modelo de Holland
V_r_squared = (numerator / denominator) + (r**2 * f_c**2) / 4       # Aplicação do modelo de Holland
V_r = np.sqrt(V_r_squared) - (r * f_c) / 2                          # Aplicação do modelo de Holland
V_r = np.nan_to_num(V_r)                                            # Aplicação do modelo de Holland

V_rot_x = -V_r * np.sin(Theta)                                      # Cálculo da coordenada x da velocidade de rotação
V_rot_y = V_r * np.cos(Theta)                                       # Cálculo da coordenada y da velocidade de rotação

V_x = V_rot_x + V_mov_x                                             # Cálculo da coordenada x da velocidade total do furacão
V_y = V_rot_y + V_mov_y                                             # Cálculo da coordenada y da velocidade total do furacão

V = np.sqrt(V_x**2 + V_y**2)                                        # Cálculo do módulo da velocidade total do furacão

plt.figure(figsize=(10, 10))
plt.imshow(V, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='viridis')
plt.quiver(X, Y, V_x, V_y, scale=Escala_Grafico)
plt.title('Campo Vetorial da Velocidade do Vento do Furacão (Modelo de Holland + Movimento)')
plt.xlabel('x (km)')
plt.ylabel('y (km)')
plt.show()

print("Raio do vórtice: ", R_max/1000, "(km)")
print("Velocidade no vórtice: {:.2f} (m/s)".format(np.max(V_r)))
print("Velocidade máxima: {:.2f} (m/s)".format(np.max(V)))