import numpy as np
import matplotlib.pyplot as plt

# Parámetros personalizables
ego_speed = 4.0
sur_speed = 2.5
dt = 0.05          # paso de tiempo (s)
total_time = 8.0

# Definir trayectorias
def ego_traj(s):
    x = -12 + 24 * s
    y = -2 + 4 * (np.sin(np.pi * (s - 0.5)) * 0.3)
    return x, y

def sur_traj(s):
    x = -2 + 4 * (np.sin(np.pi * (s - 0.5)) * 0.15)
    y = -12 + 24 * s
    return x, y

# Simulación
t = 0.0
s_ego = 0.0
s_sur = 0.0
plt.ion()  # modo interactivo
fig, ax = plt.subplots(figsize=(7,7))
ax.set_aspect('equal', 'box')
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Cruce de dos vehículos (simple)')

# Dibujar trayectorias de fondo
s = np.linspace(0, 1, 200)
x_ego, y_ego = ego_traj(s)
x_sur, y_sur = sur_traj(s)
ax.plot(x_ego, y_ego, '--', label='Trayectoria ego')
ax.plot(x_sur, y_sur, '--', label='Trayectoria surrounding')
ax.legend()

# Crear los puntos de los vehículos
ego_point, = ax.plot([], [], 'o', label='Ego')
circle1 = plt.Circle((0,0),5 , color='r')
sur_point, = ax.plot([], [], 'o', label='Surrounding')

ax.add_patch(circle1)

while t < total_time:
    s_ego = min(1.0, s_ego + ego_speed * dt / 24)  # normalizamos con longitud aprox
    s_sur = min(1.0, s_sur + sur_speed * dt / 24)
    
    x_e, y_e = ego_traj(s_ego)
    x_s, y_s = sur_traj(s_sur)
    
    ego_point.set_data([x_e], [y_e])
    sur_point.set_data([x_s], [y_s])
    circle1.set_center((x_e, y_e))
    
    plt.draw()
    plt.pause(dt)
    t += dt

plt.ioff()
plt.show()




def compute_drequired_for_trajectory(v_ego, acc_confort, priority, t_reaction, v_other): 
    if priority : 
        return 0
    else:
        return v_other * (t_reaction + v_ego / acc_confort)
    
vego_ms = 10.0/3.6
vother_ms = 50.0/3.6

dmin = compute_drequired_for_trajectory(vego_ms, 2.5, False, 0.5 , vother_ms)
print("Confort deceleration: ", dmin)

dmin = compute_drequired_for_trajectory(vego_ms, 5.0, False, 0.5 , vother_ms)
print("Emergency deceleration: ", dmin)
