import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import combinations

# --- Función de Detección de Colisiones (sin cambios) ---
def verificar_colisiones(trayectorias, dimensiones_vehiculo):
    largo, ancho = dimensiones_vehiculo
    colisiones_encontradas = []
    pares_a_verificar = list(combinations(range(len(trayectorias)), 2))
    if not pares_a_verificar: return []
    num_puntos = len(trayectorias[0])
    for i in range(num_puntos):
        for idx1, idx2 in pares_a_verificar:
            punto1, punto2 = trayectorias[idx1][i], trayectorias[idx2][i]
            s1, t1, tiempo = punto1
            s2, t2, _ = punto2
            if abs(s1 - s2) < largo and abs(t1 - t2) < ancho:
                colision = {"tiempo": tiempo, "vehiculos": (idx1 + 1, idx2 + 1)}
                colisiones_encontradas.append(colision)
    return colisiones_encontradas

# --- NUEVA FUNCIÓN PARA CALCULAR Y GRAFICAR EL PIDP ---
def calcular_y_graficar_pidp(trayectorias, distancia_seguridad):
    """
    Calcula y grafica el Perfil de Inter-Distancia Predictiva (PIDP)
    para cada par de vehículos.
    """
    pares_a_verificar = list(combinations(range(len(trayectorias)), 2))
    if not pares_a_verificar:
        print("Se necesita al menos dos vehículos para calcular el PIDP.")
        return

    # Crear una figura con subplots para cada par
    num_pares = len(pares_a_verificar)
    fig, axes = plt.subplots(num_pares, 1, figsize=(10, num_pares * 4), squeeze=False)
    fig.suptitle('Perfiles Predictivos de Inter-Distancia (PIDP)', fontsize=16)
    axes = axes.flatten() # Asegurarse de que axes sea un array 1D

    for i, (idx1, idx2) in enumerate(pares_a_verificar):
        ax = axes[i]
        puntos1 = np.array(trayectorias[idx1])
        puntos2 = np.array(trayectorias[idx2])

        # Extraer coordenadas y tiempo
        s1, t1, tiempo = puntos1.T
        s2, t2, _ = puntos2.T

        # Calcular el PIDP (distancia euclidiana en cada instante) - Vectorizado para eficiencia
        distancias = np.sqrt((s1 - s2)**2 + (t1 - t2)**2)

        # Graficar el perfil
        ax.plot(tiempo, distancias, label=f'Distancia V{idx1+1}-V{idx2+1}')
        
        # Graficar la línea de distancia de seguridad
        ax.axhline(y=distancia_seguridad, color='r', linestyle='--', label=f'Umbral de Seguridad ({distancia_seguridad}m)')

        # Encontrar y marcar la distancia mínima
        idx_min = np.argmin(distancias)
        dist_min = distancias[idx_min]
        tiempo_min = tiempo[idx_min]
        
        ax.plot(tiempo_min, dist_min, 'ro', markersize=8)
        ax.annotate(f'Mínimo: {dist_min:.2f}m',
                    xy=(tiempo_min, dist_min),
                    xytext=(tiempo_min + 0.5, dist_min + 1),
                    arrowprops=dict(facecolor='black', shrink=0.05))

        ax.set_title(f'PIDP: Vehículo {idx1+1} vs. Vehículo {idx2+1}')
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Distancia (m)')
        ax.grid(True)
        ax.legend()
        ax.set_ylim(bottom=0) # La distancia no puede ser negativa

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def analizar_trayectorias_con_pidp(trayectorias, dimensiones_vehiculo, distancia_seguridad):
    """
    Función principal que ejecuta el análisis de colisión, la visualización 3D
    y el análisis PIDP.
    """
    # 1. INFORME DE COLISIÓN
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

    # 2. ANÁLISIS PIDP
    # Esta llamada genera los gráficos de perfiles de distancia
    calcular_y_graficar_pidp(trayectorias, distancia_seguridad)

    # 3. VISUALIZACIÓN 3D CON MALLA
    fig = plt.figure(figsize=(12, 10))
    ax_3d = fig.add_subplot(111, projection='3d')
    ax_3d.set_title('Vista 3D con Volumen Espacio-Temporal')
    ax_3d.set_xlabel('Longitudinal s (m)'); ax_3d.set_ylabel('Lateral t (m)'); ax_3d.set_zlabel('Tiempo (s)')
    
    colores = ['red', 'green', 'blue']
    largo, ancho = dimensiones_vehiculo
    s_all, t_all, tiempo_all = [], [], []

    for i, trayectoria in enumerate(trayectorias):
        puntos = np.array(trayectoria)
        s, t, tiempo = puntos.T
        s_all.extend(s); t_all.extend(t); tiempo_all.extend(tiempo)
        color = colores[i % len(colores)]
        faces = []
        for j in range(len(puntos) - 1):
            s_i, t_i, time_i = puntos[j]
            s_j, t_j, time_j = puntos[j+1]
            v1,v2,v3,v4 = (s_i-largo/2,t_i-ancho/2,time_i),(s_i+largo/2,t_i-ancho/2,time_i),(s_i+largo/2,t_i+ancho/2,time_i),(s_i-largo/2,t_i+ancho/2,time_i)
            v5,v6,v7,v8 = (s_j-largo/2,t_j-ancho/2,time_j),(s_j+largo/2,t_j-ancho/2,time_j),(s_j+largo/2,t_j+ancho/2,time_j),(s_j-largo/2,t_j+ancho/2,time_j)
            verts = [v1,v2,v3,v4,v5,v6,v7,v8]
            faces.extend([[verts[0],verts[1],verts[5],verts[4]], [verts[2],verts[3],verts[7],verts[6]], [verts[0],verts[3],verts[7],verts[4]], 
                          [verts[1],verts[2],verts[6],verts[5]], [verts[0],verts[1],verts[2],verts[3]], [verts[4],verts[5],verts[6],verts[7]]])
        
        mesh = Poly3DCollection(faces, facecolor=color, edgecolor='k', alpha=0.25)
        ax_3d.add_collection3d(mesh)
        ax_3d.plot(s, t, tiempo, color=color, label=f'Vehículo {i+1}')
    
    ax_3d.legend()
    plt.show()

# --- Ejemplo de Uso (SIN CAMBIOS EN LOS VALORES) ---
if __name__ == '__main__':
    # --- PARÁMETROS DE SIMULACIÓN ---
    dims_vehiculo = (4.5, 2.0)
    DISTANCIA_SEGURIDAD = 5.0 # Umbral en metros para el PIDP

    # --- TRAYECTORIAS ---
    # Trayectoria 1
    s1_puntos = np.linspace(50, 150, 20); t1_puntos = np.full_like(s1_puntos, 0); tiempo1_puntos = np.linspace(0, 10, 20)
    trayectoria_frenet1 = list(zip(s1_puntos, t1_puntos, tiempo1_puntos))
    # Trayectoria 3
    s_puntos_3 = np.linspace(-40, 150, 20); t_puntos_3 = np.full_like(s_puntos_3, 2); tiempo_puntos_3 = np.linspace(0, 10, 20)
    trayectoria_frenet3 = list(zip(s_puntos_3, t_puntos_3, tiempo_puntos_3))
    # Trayectoria 2
    s2_puntos = np.linspace(0, 130, 20); t2_puntos = -0.25 + 3.6 / (1 + np.exp(-0.5 * (np.linspace(-5, 5, 20)))); tiempo2_puntos = np.linspace(0, 10, 20)
    trayectoria_frenet2 = list(zip(s2_puntos, t2_puntos, tiempo2_puntos))
    
    mis_trayectorias = [trayectoria_frenet1, trayectoria_frenet2, trayectoria_frenet3]
    
    analizar_trayectorias_con_pidp(mis_trayectorias, dims_vehiculo, DISTANCIA_SEGURIDAD)