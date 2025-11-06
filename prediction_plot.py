import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from itertools import combinations

def verificar_colisiones(trayectorias, dimensiones_vehiculo):
    """
    Verifica si hay colisiones entre cualquier par de vehículos a lo largo de sus trayectorias.
    """
    largo, ancho = dimensiones_vehiculo
    colisiones_encontradas = []
    
    indices_vehiculos = range(len(trayectorias))
    pares_a_verificar = list(combinations(indices_vehiculos, 2))
    
    if not pares_a_verificar:
        return []

    num_puntos = len(trayectorias[0])

    for i in range(num_puntos):
        for idx1, idx2 in pares_a_verificar:
            punto1 = trayectorias[idx1][i]
            punto2 = trayectorias[idx2][i]
            
            s1, t1, tiempo = punto1
            s2, t2, _      = punto2

            superposicion_s = abs(s1 - s2) < largo
            superposicion_t = abs(t1 - t2) < ancho
            
            if superposicion_s and superposicion_t:
                colision = {
                    "tiempo": tiempo,
                    "vehiculos": (idx1 + 1, idx2 + 1),
                    "posicion1": (s1, t1),
                    "posicion2": (s2, t2)
                }
                colisiones_encontradas.append(colision)
                
    return colisiones_encontradas

def analizar_trayectorias_frenet(trayectorias, dimensiones_vehiculo):
    """
    Visualiza trayectorias en Frenet y realiza una comprobación de colisiones.
    """
    # 1. VERIFICAR COLISIONES E INDICAR QUÉ CARROS CHOCARON
    colisiones = verificar_colisiones(trayectorias, dimensiones_vehiculo)
    print("=========================================")
    print("        INFORME DE ANÁLISIS")
    print("=========================================")
    if colisiones:
        print("--- ¡ALERTA DE COLISIÓN! ---")
        for col in colisiones:
            # ESTA LÍNEA INDICA EXACTAMENTE QUÉ VEHÍCULOS COLISIONARON
            print(f"  - Colisión detectada en tiempo={col['tiempo']:.2f}s entre Vehículo {col['vehiculos'][0]} y Vehículo {col['vehiculos'][1]}")
    else:
        print("--- Análisis de Colisión: No se encontraron colisiones. ---")
    print("=========================================\n")


    # 2. VISUALIZACIÓN 2D CON MATPLOTLIB
    fig, (ax_st, ax_stime, ax_ttime) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Análisis 2D de Trayectorias en Coordenadas de Frenet', fontsize=16)

    ax_st.set_title('Vista de Camino (s-t)')
    ax_st.set_xlabel('Coordenada Longitudinal s (m)')
    ax_st.set_ylabel('Coordenada Lateral t (m)')
    ax_st.grid(True)

    ax_stime.set_title('Avance Longitudinal (s vs Tiempo)')
    ax_stime.set_xlabel('Tiempo (s)')
    ax_stime.set_ylabel('Coordenada Longitudinal s (m)')
    ax_stime.grid(True)

    ax_ttime.set_title('Desviación Lateral (t vs Tiempo)')
    ax_ttime.set_xlabel('Tiempo (s)')
    ax_ttime.set_ylabel('Coordenada Lateral t (m)')
    ax_ttime.grid(True)

    colores = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']
    for i, trayectoria in enumerate(trayectorias):
        puntos = np.array(trayectoria)
        s_puntos, t_puntos, tiempo_puntos = puntos.T
        color = colores[i % len(colores)]
        label_v = f'Vehículo {i+1}'
        
        ax_st.plot(s_puntos, t_puntos, color=color, marker='.', linestyle='-', label=label_v)
        ax_stime.plot(tiempo_puntos, s_puntos, color=color, marker='.', linestyle='-')
        ax_ttime.plot(tiempo_puntos, t_puntos, color=color, marker='.', linestyle='-', label=label_v)
        
    # Marcar colisiones en gráficos 2D
    labeled_collision = False
    for col in colisiones:
        s_mid = (col['posicion1'][0] + col['posicion2'][0]) / 2
        t_mid = (col['posicion1'][1] + col['posicion2'][1]) / 2
        tiempo = col['tiempo']
        label = "Colisión" if not labeled_collision else ""
        ax_st.scatter(s_mid, t_mid, color='black', marker='X', s=200, zorder=10, label=label)
        ax_stime.scatter(tiempo, s_mid, color='black', marker='X', s=200, zorder=10)
        ax_ttime.scatter(tiempo, t_mid, color='black', marker='X', s=200, zorder=10)
        labeled_collision = True
        
    ax_st.legend()
    ax_ttime.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # 3. VISUALIZACIÓN 3D INTERACTIVA CON PLOTLY
    fig_3d = go.Figure()
    largo, ancho = dimensiones_vehiculo

    for i, trayectoria in enumerate(trayectorias):
        puntos = np.array(trayectoria)
        s, t, tiempo = puntos.T
        color = colores[i % len(colores)]
        fig_3d.add_trace(go.Scatter3d(x=s, y=t, z=tiempo, mode='lines',
                                     line=dict(color=color, width=4), name=f'Vehículo {i+1}'))
        
        for j in range(len(puntos) - 1):
            # ... (código para dibujar la malla 3D, sin cambios)
            s_start, t_start, time_start = puntos[j]
            s_end, t_end, time_end = puntos[j+1]
            x_verts = [s_start-largo/2, s_start+largo/2, s_start+largo/2, s_start-largo/2, s_end-largo/2, s_end+largo/2, s_end+largo/2, s_end-largo/2]
            y_verts = [t_start-ancho/2, t_start-ancho/2, t_start+ancho/2, t_start+ancho/2, t_end-ancho/2, t_end-ancho/2, t_end+ancho/2, t_end+ancho/2]
            z_verts = [time_start, time_start, time_start, time_start, time_end, time_end, time_end, time_end]
            fig_3d.add_trace(go.Mesh3d(x=x_verts, y=y_verts, z=z_verts,
                i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k=[0, 7, 2, 3, 6, 7, 2, 5, 1, 2, 7, 6],
                opacity=0.2, color=color, showlegend=False))

    # Marcar colisiones en el gráfico 3D con leyendas mejoradas
    for col in colisiones:
        s_mid = (col['posicion1'][0] + col['posicion2'][0]) / 2
        t_mid = (col['posicion1'][1] + col['posicion2'][1]) / 2
        v1, v2 = col['vehiculos']
        hover_text = f"Colisión V{v1} y V{v2}<br>Tiempo: {col['tiempo']:.2f}s<br>Posición (s,t): ({s_mid:.1f}, {t_mid:.1f})"
        
        fig_3d.add_trace(go.Scatter3d(
            x=[s_mid], y=[t_mid], z=[col['tiempo']],
            mode='markers',
            marker=dict(size=10, color='black', symbol='x'),
            name=f'Colisión V{v1} vs V{v2}', # LEYENDA MEJORADA
            hovertemplate=hover_text # TEXTO FLOTANTE AÑADIDO
        ))

    # ---- MEJORAS DE VISIBILIDAD EN PLOTLY ----
    fig_3d.update_layout(
        title='Visualización 3D Interactiva de Trayectorias (s, t, Tiempo)',
        scene=dict(
            xaxis=dict(title=dict(text='Longitudinal s (m)', font=dict(size=18))),
            yaxis=dict(title=dict(text='Lateral t (m)', font=dict(size=18))),
            zaxis=dict(title=dict(text='Tiempo (s)', font=dict(size=18)))
        ),
        legend=dict(
            font=dict(size=16),
            bgcolor='rgba(255, 255, 255, 0.7)', # Fondo semitransparente
            bordercolor='black',
            borderwidth=1
        )
    )
    fig_3d.show()


# --- Ejemplo de Uso (SIN CAMBIOS EN LOS VALORES) ---
if __name__ == '__main__':
    dims_vehiculo = (4.5, 2.0)

    # Trayectoria 1: Vehículo que se mantiene en el carril derecho (t = -1.8m)
    s1_puntos = np.linspace(50, 150, 20)
    t1_puntos = np.full_like(s1_puntos, 0)
    tiempo1_puntos = np.linspace(0, 10, 20)
    trayectoria_frenet1 = list(zip(s1_puntos, t1_puntos, tiempo1_puntos))

    # Trayectoria 2: Vehículo que se mantiene en el carril derecho (t = -1.8m)
    s_puntos_3 = np.linspace(-40, 150, 20) # Renombrada para evitar confusión
    t_puntos_3 = np.full_like(s_puntos_3, 2)
    tiempo_puntos_3 = np.linspace(0, 10, 20)
    trayectoria_frenet3 = list(zip(s_puntos_3, t_puntos_3, tiempo_puntos_3))
    
    # Trayectoria 2: Vehículo que realiza un cambio del carril derecho (t=-1.8) al izquierdo (t=1.8)
    s2_puntos = np.linspace(0, 130, 20)
    # Usamos una función sigmoide para un cambio de carril suave
    t2_puntos = -0.25 + 3.6 / (1 + np.exp(-0.5 * (np.linspace(-5, 5, 20))))
    tiempo2_puntos = np.linspace(0, 10, 20)
    trayectoria_frenet2 = list(zip(s2_puntos, t2_puntos, tiempo2_puntos))
    
    mis_trayectorias = [trayectoria_frenet1, trayectoria_frenet2, trayectoria_frenet3]
    
    analizar_trayectorias_frenet(mis_trayectorias, dims_vehiculo)