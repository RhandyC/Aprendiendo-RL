import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import combinations

def verificar_colisiones(trayectorias, dimensiones_vehiculo):
    """
    Verifica si hay colisiones entre cualquier par de vehículos a lo largo de sus trayectorias.
    (Esta función no necesita cambios).
    """
    largo, ancho = dimensiones_vehiculo
    colisiones_encontradas = []
    
    indices_vehiculos = range(len(trayectorias))
    pares_a_verificar = list(combinations(indices_vehiculos, 2))
    
    if not pares_a_verificar: return []

    num_puntos = len(trayectorias[0])

    for i in range(num_puntos):
        for idx1, idx2 in pares_a_verificar:
            punto1, punto2 = trayectorias[idx1][i], trayectorias[idx2][i]
            s1, t1, tiempo = punto1
            s2, t2, _      = punto2

            if abs(s1 - s2) < largo and abs(t1 - t2) < ancho:
                colision = {
                    "tiempo": tiempo, "vehiculos": (idx1 + 1, idx2 + 1),
                    "posicion1": (s1, t1), "posicion2": (s2, t2)
                }
                colisiones_encontradas.append(colision)
                
    return colisiones_encontradas

def analizar_trayectorias_matplotlib_mesh(trayectorias, dimensiones_vehiculo):
    """
    Visualiza trayectorias en Frenet con mallas 3D en Matplotlib y realiza
    una comprobación de colisiones.
    """
    # 1. VERIFICAR COLISIONES (sin cambios)
    colisiones = verificar_colisiones(trayectorias, dimensiones_vehiculo)
    print("=========================================")
    print("        INFORME DE ANÁLISIS")
    print("=========================================")
    if colisiones:
        print("--- ¡ALERTA DE COLISIÓN! ---")
        for col in colisiones:
            print(f"  - Colisión detectada en tiempo={col['tiempo']:.2f}s entre Vehículo {col['vehiculos'][0]} y Vehículo {col['vehiculos'][1]}")
    else:
        print("--- Análisis de Colisión: No se encontraron colisiones. ---")
    print("=========================================\n")

    # 2. CONFIGURACIÓN DE GRÁFICOS (con un subplot 3D)
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Análisis de Trayectorias con Malla 3D (Matplotlib)', fontsize=16)
    
    ax_3d = fig.add_subplot(2, 2, (1, 3), projection='3d') # El gráfico 3D ocupa la parte de arriba
    ax_st = fig.add_subplot(2, 2, 2)
    ax_ttime = fig.add_subplot(2, 2, 4)
    
    ax_3d.set_title('Vista 3D (s, t, Tiempo)')
    ax_3d.set_xlabel('Longitudinal s (m)'); ax_3d.set_ylabel('Lateral t (m)'); ax_3d.set_zlabel('Tiempo (s)')
    
    ax_st.set_title('Vista de Camino (s-t)'); ax_st.set_xlabel('s (m)'); ax_st.set_ylabel('t (m)'); ax_st.grid(True)
    ax_ttime.set_title('Desviación Lateral (t vs Tiempo)'); ax_ttime.set_xlabel('Tiempo (s)'); ax_ttime.set_ylabel('t (m)'); ax_ttime.grid(True)

    colores = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']
    largo, ancho = dimensiones_vehiculo
    
    # Listas para ajustar los límites de los ejes 3D manualmente
    s_all, t_all, tiempo_all = [], [], []

    for i, trayectoria in enumerate(trayectorias):
        puntos = np.array(trayectoria)
        s, t, tiempo = puntos.T
        s_all.extend(s); t_all.extend(t); tiempo_all.extend(tiempo)
        color = colores[i % len(colores)]
        label_v = f'Vehículo {i+1}'
        
        # Dibujar en gráficos 2D
        ax_st.plot(s, t, color=color, marker='.', linestyle='-', label=label_v)
        ax_ttime.plot(tiempo, t, color=color, marker='.', linestyle='-', label=label_v)

        # ---- CREACIÓN DE LA MALLA 3D ----
        faces = []
        for j in range(len(puntos) - 1):
            # Vértices del prisma entre el punto j y j+1
            s_i, t_i, time_i = puntos[j]
            s_j, t_j, time_j = puntos[j+1]
            
            # Vértices en el tiempo i (abajo)
            v1 = (s_i - largo/2, t_i - ancho/2, time_i)
            v2 = (s_i + largo/2, t_i - ancho/2, time_i)
            v3 = (s_i + largo/2, t_i + ancho/2, time_i)
            v4 = (s_i - largo/2, t_i + ancho/2, time_i)
            # Vértices en el tiempo j (arriba)
            v5 = (s_j - largo/2, t_j - ancho/2, time_j)
            v6 = (s_j + largo/2, t_j - ancho/2, time_j)
            v7 = (s_j + largo/2, t_j + ancho/2, time_j)
            v8 = (s_j - largo/2, t_j + ancho/2, time_j)
            
            verts = [v1, v2, v3, v4, v5, v6, v7, v8]
            
            # Definir las 6 caras del prisma
            faces.append([verts[0], verts[1], verts[5], verts[4]]) # Cara inferior
            faces.append([verts[2], verts[3], verts[7], verts[6]]) # Cara superior
            faces.append([verts[0], verts[3], verts[7], verts[4]]) # Cara lateral 1
            faces.append([verts[1], verts[2], verts[6], verts[5]]) # Cara lateral 2
            faces.append([verts[0], verts[1], verts[2], verts[3]]) # Cara "trasera" (tiempo i)
            faces.append([verts[4], verts[5], verts[6], verts[7]]) # Cara "frontal" (tiempo j)

        # Crear y añadir la colección de polígonos a la gráfica 3D
        mesh = Poly3DCollection(faces, facecolor=color, edgecolor='k', alpha=0.25)
        ax_3d.add_collection3d(mesh)
        # Trazar una línea central para la leyenda
        ax_3d.plot(s, t, tiempo, color=color, label=label_v)

    # Marcar colisiones en todos los gráficos
    for col in colisiones:
        s_mid = (col['posicion1'][0] + col['posicion2'][0]) / 2
        t_mid = (col['posicion1'][1] + col['posicion2'][1]) / 2
        tiempo = col['tiempo']
        v1, v2 = col['vehiculos']
        
        label_col = f"Colisión V{v1} vs V{v2}"
        ax_3d.scatter(s_mid, t_mid, tiempo, color='black', marker='X', s=200, label=label_col)
        ax_st.scatter(s_mid, t_mid, color='black', marker='X', s=200, zorder=10)
        ax_ttime.scatter(tiempo, t_mid, color='black', marker='X', s=200, zorder=10)

    # Ajustar límites de los ejes 3D
    ax_3d.set_xlim(min(s_all), max(s_all))
    ax_3d.set_ylim(-5, 7)
    ax_3d.set_zlim(min(tiempo_all), max(tiempo_all))
    
    ax_3d.legend()
    ax_st.legend()
    ax_ttime.legend()
    plt.tight_layout()
    plt.show()

# --- Ejemplo de Uso (SIN CAMBIOS EN LOS VALORES) ---
if __name__ == '__main__':
    dims_vehiculo = (4.5, 2.0)

    # Trayectoria 1
    s1_puntos = np.linspace(50, 150, 20)
    t1_puntos = np.full_like(s1_puntos, 0)
    tiempo1_puntos = np.linspace(0, 10, 20)
    trayectoria_frenet1 = list(zip(s1_puntos, t1_puntos, tiempo1_puntos))

    # Trayectoria 3
    s_puntos_3 = np.linspace(-40, 150, 20)
    t_puntos_3 = np.full_like(s_puntos_3, 2)
    tiempo_puntos_3 = np.linspace(0, 10, 20)
    trayectoria_frenet3 = list(zip(s_puntos_3, t_puntos_3, tiempo_puntos_3))
    
    # Trayectoria 2
    s2_puntos = np.linspace(0, 130, 20)
    t2_puntos = -0.25 + 3.6 / (1 + np.exp(-0.5 * (np.linspace(-5, 5, 20))))
    tiempo2_puntos = np.linspace(0, 10, 20)
    trayectoria_frenet2 = list(zip(s2_puntos, t2_puntos, tiempo2_puntos))
    
    mis_trayectorias = [trayectoria_frenet1, trayectoria_frenet2, trayectoria_frenet3]
    
    analizar_trayectorias_matplotlib_mesh(mis_trayectorias, dims_vehiculo)