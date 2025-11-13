import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from scipy.spatial import KDTree
from shapely.geometry import Polygon, Point
from matplotlib.patches import ConnectionPatch
from shapely.errors import TopologicalError
from shapely.geometry import LineString, Point

import math

def compute_visibility_polygon(ego_pos, angles, visibility):
    """Construye el polígono de visibilidad real a partir del ray casting"""
    points = [ego_pos + visibility[i] * np.array([np.cos(angles[i]), np.sin(angles[i])]) 
              for i in range(len(angles))]
    return Polygon(points)

def compute_required_envelope(ego_pos, trajectory, centerlanes, v_ego, acc_confort, t_reaction, v_other, priority=False):
    """
    Calcula la envolvente mínima de percepción necesaria basándose en la intersección
    de la trayectoria del ego con las centerlanes y la distancia mínima requerida.
    """
    d_required = compute_drequired_for_trajectory(v_ego, acc_confort, priority, t_reaction, v_other)
    
    ego_line = LineString(trajectory)  # tu trayectoria
    envelope_points = []

    # Para cada centerlane
    for lane in centerlanes:
        lane_line = LineString(lane)
        inter = ego_line.intersection(lane_line)
        
        if inter.is_empty:
            continue
        
        # Si es un punto de intersección
        if isinstance(inter, Point):
            envelope_points.append(np.array([inter.x, inter.y]))
            # Proyectamos hacia atras una distancia d_required en la dirección de la lane
            direction = np.array(lane[1]) - np.array(lane[0])
            direction = direction / np.linalg.norm(direction)
            projected_point = np.array([inter.x, inter.y]) - d_required * direction
            envelope_points.append(projected_point)
        
        # Si hay múltiples puntos (LineString/ MultiPoint)
        else:
            for p in inter.geoms:
                direction = np.array(lane[1]) - np.array(lane[0])
                direction = direction / np.linalg.norm(direction)
                projected_point = np.array([p.x, p.y]) + d_required * direction
                envelope_points.append(projected_point)

    return envelope_points  # lista de np.array con los puntos verdes a dibujar


def compute_visibility_coefficient(real_env, required_env):
    """Calcula el coeficiente de visibilidad"""
    intersection_area = real_env.intersection(required_env).area
    required_area = required_env.area
    return intersection_area / required_area

def compute_drequired_for_trajectory(v_ego, acc_confort, priority, t_reaction, v_other): 
    """Calcula la distancia minima respecto a una cierta aceleracion y un tiempo de reaccion"""
    if priority: 
        return 0
    else:
        return v_other * (t_reaction + v_ego / acc_confort)

class VehiclePerception:
    def __init__(self, sensor_range=100.0, weather_condition="clear"):
        self.sensor_range = sensor_range
        self.weather_condition = weather_condition
        self.weather_factors = {
            "clear": 1.0,
            "rain": 0.7,
            "fog": 0.4,
            "heavy_fog": 0.2
        }
        
    def get_effective_range(self):
        """Calcula el rango efectivo basado en condiciones climáticas"""
        return self.sensor_range * self.weather_factors[self.weather_condition]
    
    def ray_cast_visibility(self, ego_pos, obstacles, centerlanes, num_rays=360):
        """Realiza ray casting para determinar áreas visibles y ocluidas"""
        effective_range = self.get_effective_range()
        angles = np.linspace(0, 2 * np.pi, num_rays)
        visibility_map = np.zeros(num_rays)
        visibility_points = [] 

        for i, angle in enumerate(angles):
            ray_end = ego_pos + effective_range * np.array([np.cos(angle), np.sin(angle)])
            min_distance = effective_range
            intersection_point = None  # <--- nuevo
            
            for obstacle in obstacles:
                intersection_dist = self.ray_rectangle_intersection(ego_pos, ray_end, obstacle)
                if intersection_dist is not None and intersection_dist < min_distance:
                    min_distance = intersection_dist
            
            for centerlane in centerlanes:
                intersection_dist, point = self.ray_centerlane_intersection(ego_pos, ray_end, centerlane)
                if intersection_dist is not None and intersection_dist < min_distance:
                    # min_distance = intersection_dist
                    intersection_point = point 
                    
                visibility_points.append(intersection_point)
            visibility_map[i] = min_distance

        return angles, visibility_map, visibility_points
    
    def ray_rectangle_intersection(self, ray_start, ray_end, rectangle):
        """Calcula intersección entre rayo y rectángulo (bounding box)"""
        x, y, w, h = rectangle

        # Lados del rectángulo
        sides = [
            [(x, y), (x + w, y)],         # inferior
            [(x + w, y), (x + w, y + h)], # derecha
            [(x + w, y + h), (x, y + h)], # superior
            [(x, y + h), (x, y)]          # izquierda
        ]

        min_distance = None

        for side in sides:
            intersection = self.line_line_intersection(ray_start, ray_end, side[0], side[1])
            if intersection is not None:
                dist = np.linalg.norm(intersection - ray_start)
                if min_distance is None or dist < min_distance:
                    min_distance = dist

        return min_distance
    
    def ray_centerlane_intersection(self, ray_start, ray_end, centerlane):
        """Calcula intersección entre rayo y línea del carril central"""
        [init_point, final_point] = centerlane
        intersection = self.line_line_intersection(ray_start, ray_end, init_point, final_point)
        
        if intersection is not None:
            dist = np.linalg.norm(intersection - ray_start)
            return dist, intersection
        
        return None, None

    def line_line_intersection(self, A, B, C, D):
        """Calcula intersección entre dos segmentos de línea"""
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        C = np.array(C, dtype=float)
        D = np.array(D, dtype=float)
        
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        # Vectores
        r = B - A
        s = D - C
        cross_rs = r[0] * s[1] - r[1] * s[0]
        
        if abs(cross_rs) < 1e-10:
            return None  # Líneas paralelas
        
        q_minus_p = C - A
        t = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / cross_rs
        u = (q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / cross_rs
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            return A + t * r
        
        return None

class AutonomousVehicle:
    def __init__(self, initial_pos=np.array([0.0, 0.0])):
        self.position = initial_pos
        self.current_target_idx = 0
        self.speed = 8.0  # m/s
        self.perception = VehiclePerception()
        self.trajectory = [
            (0.0, 0.0),
            (20.0, 0.0),
            (40.0, 0.0),
            (60.0, 0.0),
            (80.0, 0.0),
            (100.0, 0.0)
        ]
        
    def move_along_trajectory(self, dt=0.1):
        if self.current_target_idx >= len(self.trajectory):
            return  # Ya llegó al final de la trayectoria

        # Convertir los puntos a arrays al momento de usarlos
        target = np.array(self.trajectory[self.current_target_idx])
        position = np.array(self.position)

        direction = target - position
        distance_to_target = np.linalg.norm(direction)

        if distance_to_target < 1.0:
            self.current_target_idx += 1
            return

        direction /= distance_to_target
        move_distance = self.speed * dt
        self.position += direction * move_distance

class Simulation:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.vehicle = AutonomousVehicle()
        
        # Definir bus como bounding box [x, y, width, height]
        self.bus = np.array([40.0, -2.0, 12.0, 4.0])
        self.corner1 =  np.array([-50.0, -55.0, 100.0, 50.0]) # Posición y dimensiones
        self.corner2 =  np.array([-50.0, 10.0, 100.0, 50.0]) # Posición y dimensiones

        
        initial_point = (55.0, -50.0)
        final_point = (55.0 , 50.0)
        self.up = [initial_point, final_point]

        initial_point = (52.0 , 50.0)
        final_point = (52.0 , -50.0)
        self.down = [initial_point, final_point]
        
        # np.array([[initial_point],   # punto inicial
        #                       [final_point]])  # punto final

        # Otros obstáculos (edificios, otros vehículos)
        self.obstacles = [
            self.corner1,
            self.corner2
        ]

        self.centerlanes = [
            self.up,
            self.down
        ]
        
        self.frame_count = 0
        self.visibility_data = []
        
    def update_weather_conditions(self):
        """Simula cambios en condiciones climáticas y reinicia al cambiar"""
        conditions = ["clear", "rain", "fog", "heavy_fog"]
        condition_idx = (self.frame_count // 150) % len(conditions)
        new_condition = conditions[condition_idx]
        
        # Detectar cambio de clima
        if new_condition != self.vehicle.perception.weather_condition:
            self.vehicle.perception.weather_condition = new_condition
            self.reset_simulation_state()
            
    def calculate_visibility_metrics(self, angles, visibility):
        """Calcula métricas de visibilidad"""
        effective_range = self.vehicle.perception.get_effective_range()
        visible_ratio = np.sum(visibility >= effective_range * 0.95) / len(visibility)
        avg_visible_distance = np.mean(visibility)
        
        return visible_ratio, avg_visible_distance
    
    def update(self, frame):
        self.ax.clear()
        self.frame_count = frame
        self.update_weather_conditions()
        
        # Mover vehículo hacia adelante
        self.vehicle.move_along_trajectory()

        # Calcular visibilidad
        angles, visibility, intersection_points = self.vehicle.perception.ray_cast_visibility(
            self.vehicle.position, self.obstacles, self.centerlanes
        )
        # Calcular métricas
        visible_ratio, avg_distance = self.calculate_visibility_metrics(angles, visibility)
        # self.visibility_data.append((distance_to_bus, visible_ratio, avg_distance))
        
        # Visualización
        self.plot_scenario(angles, visibility, visible_ratio, avg_distance, intersection_points)

        # Crear polígonos
        real_env = compute_visibility_polygon(self.vehicle.position, angles, visibility)
        # required_env_array = compute_required_envelope(self.vehicle.position)
        # required_env = Polygon(required_env_array)  # ahora es un Polygon

        # # Dibujar la envolvente como un polígono azul semitransparente
        # self.ax.fill(
        #     required_env_array[:, 0], required_env_array[:, 1],
        #     color='blue', alpha=0.2, label='Envolvente requerida'
        # )

        # try:
        #     C_v = compute_visibility_coefficient(real_env, required_env)
        # except TopologicalError:
        #     C_v = 0.0

        # Mostrar el coeficiente en el plot
        # self.ax.text(0.7, 0.05, f"Coef. visibilidad: {C_v:.2f}", transform=self.ax.transAxes,
        #             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        return []
    
    def plot_scenario(self, angles, visibility, visible_ratio, avg_distance, intersection_points):
        """Visualiza el escenario completo"""
        # Configurar el plot
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Dibujar obstáculos
        for obstacle in self.obstacles:
            x, y, w, h = obstacle
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor='red', facecolor='darkred', alpha=0.7)
            self.ax.add_patch(rect)
        
        # Dibujar centerlanes
        for centerlane in self.centerlanes:
            [init_point, final_point] = centerlane
            # self.ax.plot(   [init_point[0], final_point[0]],
            #                 [init_point[1], final_point[1]],
            #                 color='blue',
            #                 linewidth=2,
            #                 label='Centerlane')
        
            dx = final_point[0] - init_point[0]
            dy = final_point[1] - init_point[1]

            self.ax.arrow(init_point[0], init_point[1], dx, dy, head_width=2, head_length=8, fc='blue', ec='blue')

            
        for p in intersection_points:
            if p is not None:
                self.ax.plot(p[0], p[1], 'ro', markersize=4) 

        # Calcular puntos requeridos según la fórmula
        d_required_points = compute_required_envelope(self.vehicle.position,
                                                    self.vehicle.trajectory,
                                                    self.centerlanes,
                                                    v_ego=self.vehicle.speed,
                                                    acc_confort=2.5,
                                                    t_reaction=0.5,
                                                    v_other=50.0/3.6,
                                                    priority=False)
        # Dibujar puntos verdes proyectados sobre las lanes
        for p in d_required_points:
            self.ax.plot(p[0], p[1], 'go', markersize=5)  # 'go' = green circle

        # Dibujar vehículo autónomo
        vehicle_circle = patches.Circle(self.vehicle.position, radius=1.5, 
                                      facecolor='blue', edgecolor='darkblue', linewidth=2)
        self.ax.add_patch(vehicle_circle)
        
        # Dibujar envolvente de percepción
        effective_range = self.vehicle.perception.get_effective_range()
        
        # Área completamente visible
        visible_circle = patches.Circle(self.vehicle.position, effective_range, 
                                      fill=False, edgecolor='green', linestyle='--', alpha=0.5)
        self.ax.add_patch(visible_circle)
        
        # Visualizar rayos (solo algunos para no saturar)
        num_show_rays = 100
        step = len(angles) // num_show_rays
        
        for i in range(0, len(angles), step):
            angle = angles[i]
            dist = visibility[i]
            end_point = self.vehicle.position + dist * np.array([np.cos(angle), np.sin(angle)])
            
            if dist >= effective_range * 0.95:
                color = 'green'
                alpha = 0.3
            else:
                color = 'orange'
                alpha = 0.6
                
            self.ax.plot([self.vehicle.position[0], end_point[0]], 
                        [self.vehicle.position[1], end_point[1]], 
                        color=color, alpha=alpha, linewidth=1)
        
        # Información del estado
        info_text = (f"Frame: {self.frame_count}\n"
                    f"Condición: {self.vehicle.perception.weather_condition}\n"
                    f"Rango efectivo: {effective_range:.1f}m\n"
                    f"Área visible: {visible_ratio*100:.1f}%\n"
                    f"Dist. visible prom: {avg_distance:.1f}m")
        
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.ax.set_title(f"Evolución de Envolvente de Percepción - Frame {self.frame_count}")

    def reset_simulation_state(self):
        """Reinicia la posición y datos de la simulación"""
        self.vehicle.position = np.array([0.0, 0.0])
        self.visibility_data.clear()
        print(f"--- Reiniciando simulación por cambio climático: {self.vehicle.perception.weather_condition} ---")


def plot_visibility_evolution(simulation):
    """Grafica la evolución de la visibilidad a lo largo del tiempo"""
    if len(simulation.visibility_data) == 0:
        return
    
    distances = [data[0] for data in simulation.visibility_data]
    visible_ratios = [data[1] * 100 for data in simulation.visibility_data]  # Porcentaje
    avg_distances = [data[2] for data in simulation.visibility_data]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Gráfico de área visible vs distancia
    ax1.plot(distances, visible_ratios, 'b-', linewidth=2, label='Área Visible (%)')
    ax1.set_xlabel('Distancia al Bus (m)')
    ax1.set_ylabel('Porcentaje de Área Visible (%)')
    ax1.set_title('Evolución del Área Visible vs Distancia al Bus')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.invert_xaxis()  # Porque la distancia disminuye
    
    # Gráfico de distancia visible promedio
    ax2.plot(distances, avg_distances, 'g-', linewidth=2, label='Distancia Visible Promedio')
    ax2.axhline(y=simulation.vehicle.perception.sensor_range, color='r', linestyle='--', 
                label='Rango Máximo del Sensor')
    ax2.set_xlabel('Distancia al Bus (m)')
    ax2.set_ylabel('Distancia Visible (m)')
    ax2.set_title('Evolución de la Distancia Visible Promedio')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.invert_xaxis()
    
    plt.tight_layout()
    plt.show()

# Ejecutar simulación
print("Iniciando simulación de evolución de envolvente de percepción...")
sim = Simulation()

# Crear animación
animation = FuncAnimation(sim.fig, sim.update, frames=240, interval=50, blit=False)

plt.show()

# Al finalizar, mostrar gráficos de evolución
plot_visibility_evolution(sim)