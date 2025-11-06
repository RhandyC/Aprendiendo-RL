import gymnasium as gym
import numpy as np

# Inicia con un espacio de observación complejo
env = gym.make(
    'FrozenLake-v1',
    desc=None,
    map_name="4x4",
    is_slippery=False,
    render_mode="human"
)
gamma = 0.99
theta = 0.000001

def argmax(env, V, pi, action, s, gamma):
    e = np.zeros(env.action_space.n)
    for a in range(env.action_space.n):                         # itera para cada acción posible
        q = 0
        # CAMBIO AQUÍ: Usa env.unwrapped.P para acceder a la matriz de transición
        P = np.array(env.unwrapped.P[s][a])
        (x, y) = np.shape(P)                             # para la Ecuación de Bellman

        for i in range(x):                              # itera para cada estado posible
            s_ = int(P[i][1])                            # S' - Sprime - posibles estados sucesores
            p = P[i][0]                                 # Probabilidad de transición P(s'|s,a)
            r = P[i][2]                                 # Recompensa

            q += p * (r + gamma * V[s_])                      # calcula el valor de la acción q(s|a)
            e[a] = q

    m = np.argmax(e)
    action[s] = m                                           # Toma el índice que tiene el valor máximo
    pi[s][m] = 1                                        # actualiza pi(a|s)

    return pi


def bellman_optimality_update(env, V, s, gamma):  # actualiza el valor de estado V[s] tomando
    pi = np.zeros((env.observation_space.n, env.action_space.n))       # la acción que maximiza el valor actual
    e = np.zeros(env.action_space.n)
                                            # PASO 1: Encontrar
    for a in range(env.action_space.n):
        q = 0                                 # itera para todas las acciones posibles
        # CAMBIO AQUÍ: Usa env.unwrapped.P para acceder a la matriz de transición
        P = np.array(env.unwrapped.P[s][a])
        (x, y) = np.shape(P)

        for i in range(x):
            s_ = int(P[i][1])
            p = P[i][0]
            r = P[i][2]
            q += p * (r + gamma * V[s_])
            e[a] = q

    m = np.argmax(e)
    pi[s][m] = 1

    value = 0
    for a in range(env.action_space.n):
        u = 0
        # CAMBIO AQUÍ: Usa env.unwrapped.P para acceder a la matriz de transición
        P = np.array(env.unwrapped.P[s][a])
        (x, y) = np.shape(P)
        for i in range(x):

            s_ = int(P[i][1])
            p = P[i][0]
            r = P[i][2]

            u += p * (r + gamma * V[s_])

        value += pi[s, a] * u

    V[s] = value
    return V[s]



def value_iteration(env, gamma, theta):
    V = np.zeros(env.observation_space.n)                                       # inicializa v(0) a un valor arbitrario, en mi caso "ceros"
    while True:
        delta = 0
        for s in range(env.observation_space.n):                       # itera para todos los estados
            v = V[s]
            bellman_optimality_update(env, V, s, gamma)   # actualiza el valor de estado con bellman_optimality_update
            delta = max(delta, abs(v - V[s]))             # asigna el cambio en el valor por iteración a delta
        if delta < theta:
            break                                         # si el cambio se vuelve insignificante
                                                          # --> converge al valor óptimo
    pi = np.zeros((env.observation_space.n, env.action_space.n))
    action = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        pi = argmax(env, V, pi, action, s, gamma)         # extrae la política óptima usando el valor de la acción

    return V, pi, action                                          # función de valor óptima, política óptima


V, pi, action = value_iteration(env, gamma, theta)

a = np.reshape(action, (4, 4))
print("Política Óptima (0: Izquierda, 1: Abajo, 2: Derecha, 3: Arriba):")
print(a)                          # acción discreta a tomar en un estado dado


e = 0
for i_episode in range(5):
    # env.reset() ahora devuelve una tupla (estado, info)
    c, info = env.reset()
    for t in range(10000):
        # env.step() ahora devuelve (estado, recompensa, terminado, truncado, info)
        c, reward, terminated, truncated, info = env.step(int(action[c]))
        done = terminated or truncated # El episodio termina si es 'terminado' O 'truncado'
        if done:
            if reward == 1:
                e += 1
            break
print(f"El agente logró alcanzar la meta {e} de 100 episodios usando esta política.")
env.close()