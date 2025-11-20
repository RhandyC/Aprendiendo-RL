import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.errors import TopologicalError
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math


class AutonomousVehicle:
    def __init__(self, initial_pos=np.array([0.0, 0.0])):
        self.position = initial_pos
        self.current_target_idx = 0
        self.speed = 6.0  # m/s
        self.perception = VehiclePerception()
        self.trajectory = [
            (0.0, 0.0),
            (20.0, 0.0),
            (40.0, 0.0),
            (60.0, 0.0),
            (80.0, 0.0),
            (100.0, 0.0)
        ]

        self.length = 4.0    
        self.width = 1.8   
        self.acc_confort = 2.5
        self.t_reaction = 0.5
        
    def move_along_trajectory(self, dt=0.1):
        if self.current_target_idx >= len(self.trajectory):
            return  # Already reached the end of the trajectory

        # Convert points to arrays when using them
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
    
    def redefined_trajectory(self, trajectory): 
        self.trajectory = trajectory
        self.position = trajectory[0]
    
    def set_speed(self,speed):
        self.speed = speed

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
        """Calculates the effective range based on weather conditions"""
        return self.sensor_range * self.weather_factors[self.weather_condition]
    
    def ray_cast_visibility(self, ego_pos, obstacles, centerlanes, num_rays=360):
        """Performs ray casting to determine visible and occluded areas"""
        effective_range = self.get_effective_range()
        angles = np.linspace(0, 2 * np.pi, num_rays)
        visibility_map = np.zeros(num_rays)
        visibility_points = [] 

        for i, angle in enumerate(angles):
            ray_end = ego_pos + effective_range * np.array([np.cos(angle), np.sin(angle)])
            min_distance = effective_range
            intersection_point = None  # <--- new
            
            for obstacle in obstacles:
                # intersection_dist = self.ray_rectangle_intersection(ego_pos, ray_end, obstacle)
                intersection_dist = self.ray_polygon_intersection(ego_pos, ray_end, obstacle)
                if intersection_dist is not None and intersection_dist < min_distance:
                    min_distance = intersection_dist
            
            for centerlane in centerlanes:
                intersection_dist, point = self.ray_centerlane_intersection(ego_pos, ray_end, centerlane.segment)
                if intersection_dist is not None and intersection_dist < min_distance:
                    # min_distance = intersection_dist
                    intersection_point = point 
                    visibility_points.append(intersection_point)
            visibility_map[i] = min_distance

        return angles, visibility_map, visibility_points

    def ray_cast_visibility_simplified(self, ego_pos, obstacles, centerlanes):
        """Performs ray casting to determine visible and occluded areas considering only bounding boxes and relative position"""
        slopes = []

        for x, y in obstacles.vertices:
            if x == 0:
                slopes.append(float('inf') if y > 0 else -float('inf'))
            else:
                slopes.append(y / x)

        # return min(slopes), max(slopes)
    
        angle_min = np.arctan2(min(slopes), 1.0)
        angle_max = np.arctan2(max(slopes), 1.0)

        angles = [angle_min,angle_max]
        effective_range = self.get_effective_range()
        visibility_points = [] 

        ray_end = ego_pos + effective_range * np.array([np.cos(angles), np.sin(angles)])
        # To do: manage multiples oclusions 
        return angles, visibility_points
    
    def ray_rectangle_intersection(self, ray_start, ray_end, rectangle):
        """Calculates intersection between ray and rectangle (bounding box)"""
        x, y, w, h = rectangle

        # Sides of the rectangle
        sides = [
            [(x, y), (x + w, y)],         # bottom
            [(x + w, y), (x + w, y + h)], # right
            [(x + w, y + h), (x, y + h)], # top
            [(x, y + h), (x, y)]          # left
        ]

        min_distance = None

        for side in sides:
            intersection = self.line_line_intersection(ray_start, ray_end, side[0], side[1])
            if intersection is not None:
                dist = np.linalg.norm(intersection - ray_start)
                if min_distance is None or dist < min_distance:
                    min_distance = dist

        return min_distance
    
    def ray_polygon_intersection(self, ray_start, ray_end, polygon_vertices):
        """Calculates the closest intersection between a ray and a polygon's edges."""
        min_distance = None

        # Iterar sobre cada lado del polígono
        for i in range(len(polygon_vertices)):
            # El lado del polígono va desde el vértice i al vértice i+1
            p1 = polygon_vertices[i]
            p2 = polygon_vertices[(i + 1) % len(polygon_vertices)] # El % asegura que el último vértice conecte con el primero

            # Comprobar la intersección del rayo con este lado
            intersection_point = self.line_line_intersection(ray_start, ray_end, p1, p2)

            if intersection_point is not None:
                # Si hay intersección, calcular la distancia
                dist = np.linalg.norm(intersection_point - ray_start)
                
                # Quedarse con la distancia más corta
                if min_distance is None or dist < min_distance:
                    min_distance = dist
        
        return min_distance
    
    def ray_centerlane_intersection(self, ray_start, ray_end, centerlane):
        """Calculates intersection between ray and center lane line"""
        [init_point, final_point] = centerlane
        intersection = self.line_line_intersection(ray_start, ray_end, init_point, final_point)
        
        if intersection is not None:
            dist = np.linalg.norm(intersection - ray_start)
            return dist, intersection
        
        return None, None

    def line_line_intersection(self, A, B, C, D):
        """Calculates intersection between two line segments"""
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        C = np.array(C, dtype=float)
        D = np.array(D, dtype=float)
        
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        # Vectors
        r = B - A
        s = D - C
        cross_rs = r[0] * s[1] - r[1] * s[0]
        
        if abs(cross_rs) < 1e-10:
            return None  # Parallel lines
        
        q_minus_p = C - A
        t = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / cross_rs
        u = (q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / cross_rs
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            return A + t * r
        
        return None

class Obstacle:
    def __init__(self, center, length, width):
        cx, cy = center
        L = length / 2
        W = width / 2

        self.vertices = [
            (cx - L, cy - W),  # bottom-left
            (cx - L, cy + W),  # top-left
            (cx + L, cy + W),  # top-right
            (cx + L, cy - W)   # bottom-right
        ]

class Centerlane:
    def __init__(self, initial_point, final_point, speed_obstacle = 10.0):
        self.segment = [initial_point, final_point] # relative to ego
        self.speed_obstacle = speed_obstacle # m/s

class Simulation:
    def __init__(self, usecase):
        if usecase == "empty":
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.vehicle = AutonomousVehicle()
            self.speed_obstacle = 0.0
            # Other obstacles (buildings, other vehicles)
            self.obstacles = [
            ]

            self.centerlanes = [
            ]
            self.frame_count = 0
            self.visibility_data = []

        elif usecase == "angle_joint": 
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.vehicle = AutonomousVehicle()
            self.speed_obstacle = 8.0
            # New irregular obstacle
            angle = np.pi/5
            ego_lane_width = 7.5
            new_obstacle_poly1 = [
                (-100.0, -ego_lane_width/2), 
                (50.0, -ego_lane_width/2), 
                (50.0 - 50/math.tan(angle), -50.0), 
                (-100.
                , -50.0)
            ]
            new_obstacle_poly2 = [
                (-100.0, ego_lane_width/2), 
                (50.0 + ego_lane_width/math.tan(angle), ego_lane_width/2), 
                (50.0 + (ego_lane_width+50.0)/math.tan(angle), 50.0), 
                (-100.0, 50.0)
            ]
            new_obstacle_poly3 = [
                (50.0 + (ego_lane_width + 50.0 + 2*ego_lane_width)/math.tan(angle), 50.0), 
                (50.0 - 50/math.tan(angle) + 2*ego_lane_width/math.tan(angle), -50.0),
                (100.0, -50.0), 
                (250.0, 50.0)
            ]

            initial_point = (50.0 + ((ego_lane_width*3/2)-50.0)/math.tan(angle), -50.0)
            final_point = (50.0 - 50/math.tan(angle) + (100+ego_lane_width+ ego_lane_width*3/2)/math.tan(angle), 50.0)
            up = Centerlane (initial_point, final_point, self.speed_obstacle)

            initial_point = (50.0 + (ego_lane_width+50.0+(ego_lane_width*1/2))/math.tan(angle), 50.0)
            final_point = (50.0 + ((ego_lane_width*1/2) - 50.0)/math.tan(angle), -50.0)
            down = Centerlane (initial_point, final_point, self.speed_obstacle)

            # Other obstacles (buildings, other vehicles)
            self.obstacles = [
                new_obstacle_poly1,
                new_obstacle_poly2,
                new_obstacle_poly3 
            ]

            self.centerlanes = [
                up,
                down
            ]

            self.frame_count = 0
            self.visibility_data = []
        
        elif usecase == "bus_overtaking": 
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.vehicle = AutonomousVehicle()
            self.speed_obstacle = 2.0
            # New bus obstacle
            angle = np.pi/3
            ego_lane_width = 7.5
            self.bus = [
                (20.0, -ego_lane_width/2), 
                (32.0, -ego_lane_width/2),
                (32.0, -ego_lane_width/2 - 4.0),
                (20.0, -ego_lane_width/2 - 4.0), 
            ]

            initial_point = (34.0, -18.0)
            final_point = (34.0, 15.0)
            pedestrian = Centerlane (initial_point, final_point, self.speed_obstacle)
            
            # np.array([[initial_point],   # initial point
            #                       [final_point]])  # final point

            # Other obstacles (buildings, other vehicles)
            self.obstacles = [
                self.bus
            ]

            self.centerlanes = [
                pedestrian
            ]
            
            self.frame_count = 0
            self.visibility_data = []
        
        elif usecase == "lane_change_highway": 
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.vehicle = AutonomousVehicle()
            self.speed_obstacle = 36.0
            lane_change = [
                (0.0, 0.0),
                (0.0, 2.0),
                (0.0, 2.5),
                (0.0, 3.0),
                (0.0, 4.0),
                (0.0, 5.0)
            ] 

            self.vehicle.redefined_trajectory(lane_change)

            # New truck obstacle
            angle = np.pi/3
            ego_lane_width = 7.5
            car_front = [
                (20.0, 1.0), 
                (25.0, 2.0),
                (25.0, 2.0),
                (20.0, 1.0), 
            ]

            car_rear = [
                (-20.0 -10, -1.4), 
                (-10.0-10, -1.4),
                (-10.0 -10, 1.4),
                (-20-10 , 1.4), 
            ]

            initial_point = (-50.0, 3.5)
            final_point = (50.0, 3.5)
            left_lane = Centerlane (initial_point, final_point, self.speed_obstacle)

            # Other obstacles (buildings, other vehicles)
            self.obstacles = [
                # self.car_front,
                car_rear
            ]

            self.centerlanes = [
                left_lane
            ]
            
            self.frame_count = 0
            self.visibility_data = []

        elif usecase == "t_joint": 
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.vehicle = AutonomousVehicle()
            self.speed_obstacle = 8.0
            # Obstacles defined as 4-edge polygones 
            corner1_poly = rect_to_poly([-50.0, -55.0, 100.0, 50.0])
            corner2_poly = rect_to_poly([-50.0, 10.0, 100.0, 50.0])

            # New irregular obstacle
            self.new_obstacle_poly = [
                (30.0, -10.0), 
                (50.0, -15.0), 
                (55.0, 5.0), 
                (35.0, 8.0)
            ]
            initial_point = (55.0, -50.0)
            final_point = (55.0 , 50.0)
            up = Centerlane (initial_point, final_point, self.speed_obstacle)

            initial_point = (52.0 , 50.0)
            final_point = (52.0 , -50.0)
            down = Centerlane (initial_point, final_point, self.speed_obstacle)
            
            # np.array([[initial_point],   # initial point
            #                       [final_point]])  # final point

            # Other obstacles (buildings, other vehicles)
            self.obstacles = [
                corner1_poly,
                corner2_poly,
            ]

            self.centerlanes = [
                up,
                down
            ]
            
            self.frame_count = 0
            self.visibility_data = []

        elif usecase == "roundbout": 
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.vehicle = AutonomousVehicle()
            self.speed_obstacle = 8.0

            roundabout_poly = create_roundabout(center=(50, 0), radius=15)

            # Other obstacles (buildings, other vehicles)
            self.obstacles = [
                roundabout_poly,
            ]

            self.centerlanes = [
            ]

            self.frame_count = 0
            self.visibility_data = []

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def add_centerlane(self, centerlane):
        self.centerlanes.append(centerlane)

    def set_trajectory(self, trajectory):
        self.vehicle.redefined_trajectory(trajectory)

    def ego_set_speed(self, speed):
        self.vehicle.set_speed(speed)
        
    def update_weather_conditions(self):
        """Simulates changes in weather conditions and resets upon change"""
        conditions = ["clear", "rain", "fog", "heavy_fog"]
        condition_idx = (self.frame_count // 150)

        if condition_idx >= len(conditions):
            plt.close(self.fig)  # Cierra la ventana de la animación
            return  # 🔴 Importante: salir de la función aquí

        # Si no cerramos, sí definimos la nueva condición
        new_condition = conditions[condition_idx]

        # Detect weather change
        if new_condition != self.vehicle.perception.weather_condition:
            self.vehicle.perception.weather_condition = new_condition
            self.reset_simulation_state()
            
    def calculate_visibility_metrics(self, angles, visibility):
        """Calculates visibility metrics"""
        effective_range = self.vehicle.perception.get_effective_range()
        visible_ratio = np.sum(visibility >= effective_range * 0.95) / len(visibility)
        avg_visible_distance = np.mean(visibility)
        
        return visible_ratio, avg_visible_distance
    
    def update(self, frame):
        self.ax.clear()
        self.frame_count = frame
        self.update_weather_conditions()
        
        # Move vehicle forward
        self.vehicle.move_along_trajectory()

        # Calculate visibility
        angles, visibility, intersection_points = self.vehicle.perception.ray_cast_visibility(
            self.vehicle.position, self.obstacles, self.centerlanes
        )
        # Calculate metrics
        visible_ratio, avg_distance = self.calculate_visibility_metrics(angles, visibility)
        # self.visibility_data.append((distance_to_bus, visible_ratio, avg_distance))
        
        # Visualization
        self.plot_scenario(angles, visibility, visible_ratio, avg_distance, intersection_points)

        # Create polygons
        real_env = compute_visibility_polygon(self.vehicle.position, angles, visibility)
        text_lines = []

        for idx, centerlane in enumerate(self.centerlanes):
            required_env = compute_required_envelope(
                self.vehicle.position,
                self.vehicle.trajectory,
                centerlane.segment,
                v_ego=self.vehicle.speed,
                acc_confort=self.vehicle.acc_confort,
                t_reaction=self.vehicle.t_reaction,
                v_other=centerlane.speed_obstacle,
                priority=False
            )
            
            # Draw green points projected onto the lanes
            for p in required_env:
                self.ax.plot(p[0], p[1], 'go', markersize=5)
            
            # Calculate visibility coefficient
            try:
                C_v, (inter_init_point, inter_last_point) = compute_visibility_coefficient(
                    intersection_points, required_env, centerlane.segment
                )
            except TopologicalError:
                C_v = 0.0
                inter_init_point = inter_last_point = None

            # Draw intersection segment if it exists
            if inter_init_point is not None and inter_last_point is not None:
                self.ax.plot(
                    [inter_init_point[0], inter_last_point[0]],
                    [inter_init_point[1], inter_last_point[1]],
                    'purple', markersize=5, linewidth=2, label=f'Intersection lane {idx+1}'
                )
            
            # Save text for each lane
            text_lines.append(f"Lane {idx+1}: C_v = {C_v:.2f}")

        # Display all coefficients in the plot
        text_str = "\n".join(text_lines)
        self.ax.text(
            0.7, 0.05, text_str,
            transform=self.ax.transAxes,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        )

        # Display ego_trajectory
        x, y = zip(*self.vehicle.trajectory)
        # Graficar con línea negra y puntos visibles
        self.ax.plot(x, y, 'ko-', markersize=2)

        return []
    
    def plot_scenario(self, angles, visibility, visible_ratio, avg_distance, intersection_points):
        """Visualizes the complete scenario"""
        # Configure the plot
        # self.ax.set_xlim(-100, 100)
        # self.ax.set_ylim(-100, 100)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Draw obstacles
        for obstacle_vertices in self.obstacles:
            poly_patch = patches.Polygon(obstacle_vertices, closed=True, linewidth=2,
                                 edgecolor='red', facecolor='darkred', alpha=0.7)
            self.ax.add_patch(poly_patch)
        
        # Draw centerlanes
        for idx, centerlane in enumerate(self.centerlanes):
            init_point, final_point = centerlane.segment
            dx = final_point[0] - init_point[0]
            dy = final_point[1] - init_point[1]

            # Draw the arrow
            self.ax.arrow(
                init_point[0], init_point[1],
                dx, dy,
                head_width=2, head_length=8,
                fc='blue', ec='blue',
                linewidth=2
            )

            self.ax.text(
                init_point[0], init_point[1],
                f"Lane {idx+1}",
                color='black',
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='bottom'
            )
            
        for p in intersection_points:
            if p is not None:
                self.ax.plot(p[0], p[1], 'ro', markersize=4) 

        # Draw autonomous vehicle
        bottom_left = (self.vehicle.position[0] - self.vehicle.length / 2, self.vehicle.position[1] - self.vehicle.width / 2)

        vehicle_rect = patches.Rectangle(
                            bottom_left,
                            self.vehicle.length,
                            self.vehicle.width,
                            linewidth=0.5,
                            edgecolor='black',
                            facecolor='green',
                            zorder=10  
                        )
        self.ax.add_patch(vehicle_rect)
        
        # Draw perception envelope
        effective_range = self.vehicle.perception.get_effective_range()
        
        # Completely visible area
        visible_circle = patches.Circle(self.vehicle.position, effective_range, 
                                      fill=False, edgecolor='green', linestyle='--', alpha=0.5)
        self.ax.add_patch(visible_circle)
        
        # Visualize rays (only some to avoid clutter)
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
        
        # Status information
        info_text = (f"Frame: {self.frame_count}\n"
                    f"Condition: {self.vehicle.perception.weather_condition}\n"
                    f"Effective Range: {effective_range:.1f}m\n"
                    f"Visible Area: {visible_ratio*100:.1f}%\n")
        
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.ax.set_title(f"Evolution of Perception Envelope - Frame {self.frame_count}")

    def reset_simulation_state(self):
        """Resets the simulation's position and data"""
        self.vehicle.position = np.array([0.0, 0.0])
        self.visibility_data.clear()
        print(f"--- Resetting simulation due to weather change: {self.vehicle.perception.weather_condition} ---")

    def plot_frame(self, target_frame): 
        # Asegurarse de que el vehículo y el estado de la simulación comiencen desde el inicio
        # para que cada llamada a plot_frame sea independiente.

        self.vehicle.position = self.vehicle.trajectory[0]
        self.vehicle.current_target_idx = 0 # Reiniciar el índice del objetivo
        self.frame_count = 0 # Reiniciar el contador de frames
        self.vehicle.perception.weather_condition = "clear" # Reiniciar condición climática

        # Avanzar la simulación hasta el frame deseado
        for i in range(target_frame):
            self.frame_count = i # Actualizar el frame_count para update_weather_conditions
            self.update_weather_conditions()
            self.vehicle.move_along_trajectory()
        
        # Una vez que el vehículo ha avanzado hasta el target_frame, dibujar la escena
        self.ax.clear()
        self.frame_count = target_frame # Asegurarse de que el frame_count refleje el frame actual para el título y la info

        # Calculate visibility (needed for plotting the perception envelope)
        angles, visibility, intersection_points = self.vehicle.perception.ray_cast_visibility(
            self.vehicle.position, self.obstacles, self.centerlanes
        )
        # Calculate metrics (also for display in plot_scenario)
        visible_ratio, avg_distance = self.calculate_visibility_metrics(angles, visibility)
        
        # Call plot_scenario to draw everything for the current frame
        self.plot_scenario(angles, visibility, visible_ratio, avg_distance, intersection_points)

        # Create polygons (for required envelope and visibility coefficient)
        real_env = compute_visibility_polygon(self.vehicle.position, angles, visibility)
        text_lines = []

        for idx, centerlane in enumerate(self.centerlanes):
            required_env = compute_required_envelope(
                self.vehicle.position,
                self.vehicle.trajectory,
                centerlane.segment,
                v_ego=self.vehicle.speed,
                acc_confort=self.vehicle.acc_confort,
                t_reaction=self.vehicle.t_reaction,
                v_other=centerlane.speed_obstacle,
                priority=False
            )
            
            # Draw green points projected onto the lanes
            for p in required_env:
                self.ax.plot(p[0], p[1], 'go', markersize=5)
            
            # Calculate visibility coefficient
            try:
                C_v, (inter_init_point, inter_last_point) = compute_visibility_coefficient(
                    intersection_points, required_env, centerlane.segment
                )
            except TopologicalError:
                C_v = 0.0
                inter_init_point = inter_last_point = None

            # Draw intersection segment if it exists
            if inter_init_point is not None and inter_last_point is not None:
                self.ax.plot(
                    [inter_init_point[0], inter_last_point[0]],
                    [inter_init_point[1], inter_last_point[1]],
                    'purple', markersize=5, linewidth=2, label=f'Intersection lane {idx+1}'
                )
            
            # Save text for each lane
            text_lines.append(f"Lane {idx+1}: C_v = {C_v:.2f}")

        # Display all coefficients in the plot
        text_str = "\n".join(text_lines)
        self.ax.text(
            0.7, 0.05, text_str,
            transform=self.ax.transAxes,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        )

        # Display ego_trajectory
        x, y = zip(*self.vehicle.trajectory)
        self.ax.plot(x, y, 'ko-', markersize=2)

        plt.show() # Make sure to show the plot for the single frame



def rect_to_poly(rect):
    x, y, w, h = rect
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

def compute_visibility_coefficient(real_env, required_env, centerlane, margin=4.0):
    """Calculates the visibility coefficient"""

    real_st = [global_to_st(p, centerlane) for p in real_env]
    # Filter real_env points: keep only those close to the centerlane (t ≈ 0)
    t_threshold = 0.2  # ajusta si quieres
    real_st = [(s, t) for (s, t) in real_st if abs(t) <= t_threshold]

    required_st = [global_to_st(p, centerlane) for p in required_env]

    # Extract only the longitudinal s-axis
    real_s = [s for s, _ in real_st]
    required_s = [s for s, _ in required_st]

    # If there are no points in real_s or required_s, there is no intersection
    if len(real_s) == 0 or len(required_s) == 0:
        return 0.0, (None, None)

    # Calculate overlapping intervals in s
    sA_min, sA_max = np.min(required_s), np.max(required_s) # Required distance to see

     # ---- APPLY MARGIN ----
    sA_min_expanded = sA_min - margin
    sA_max_expanded = sA_max + margin

    # ---- Extract real_s values inside expanded interval ----
    real_inside = [s for s in real_s if sA_min_expanded <= s <= sA_max_expanded]

    if len(real_inside) == 0:
        # No part of the real segment lies in the expanded required range
        return 0.0, (None, None)

    sB_inside_min = np.min(real_inside)
    sB_inside_max = np.max(real_inside)

    inter_min = max(sA_min, sB_inside_min)
    inter_max = min(sA_max, sB_inside_max)

    if inter_max <= inter_min:
        intersection_length = 0.0
    else:
        intersection_length = inter_max - inter_min

    length_A = sA_max - sA_min
    ratio = intersection_length / length_A if length_A > 0 else 0.0

    if ratio > 0:
        inter_init_point = st_to_global(inter_min, 0, centerlane)
        inter_last_point = st_to_global(inter_max, 0, centerlane)
    else:
        inter_init_point, inter_last_point = None, None

    return ratio, (inter_init_point, inter_last_point)

def global_to_st(point, centerlane):
    """
    Converts a point (x,y) in global coordinates to the local (s,t) system
    defined by the first two points of centerlane.
    """
    origin = np.array(centerlane[0])
    direction = np.array(centerlane[1]) - origin
    direction = direction / np.linalg.norm(direction)  # normalize

    # perpendicular vector (90° counter-clockwise rotation)
    perp = np.array([-direction[1], direction[0]])

    # global->local rotation matrix
    R = np.column_stack((direction, perp))  # [s_axis | t_axis]

    # vector from origin to the point
    vec = np.array(point) - origin

    # local coordinates (s, t)
    local = R.T @ vec
    s, t = local[0], local[1]
    return s, t

def st_to_global(s, t, centerlane):
    origin = np.array(centerlane[0])
    direction = np.array(centerlane[1]) - origin
    direction = direction / np.linalg.norm(direction)
    perp = np.array([-direction[1], direction[0]])
    R = np.column_stack((direction, perp))
    return origin + R @ np.array([s, t])

def compute_required_envelope(ego_pos, trajectory, centerlane, v_ego, acc_confort, t_reaction, v_other, priority=False):
    """
    Calculates the minimum required perception envelope based on the intersection
    of the ego's trajectory with the centerlanes and the required minimum distance.
    """
    d_required = compute_drequired_for_trajectory(v_ego, acc_confort, priority, t_reaction, v_other)
    ego_line = LineString(trajectory)  # your trajectory
    envelope_points = []

    lane_line = LineString(centerlane)
    inter = ego_line.intersection(lane_line)
    
    # If it's an intersection point
    if not isinstance(inter,Point):
        return envelope_points 

    if isinstance(inter, Point):
        envelope_points.append(np.array([inter.x, inter.y]))
        # We project backwards a distance d_required in the direction of the lane
        direction = np.array(centerlane[1]) - np.array(centerlane[0])
        direction = direction / np.linalg.norm(direction)
        projected_point = np.array([inter.x, inter.y]) - d_required * direction
        envelope_points.append(projected_point)
    
    # If there are multiple points (LineString/ MultiPoint)
    else:
        for p in inter.geoms:
            direction = np.array(centerlane[1]) - np.array(centerlane[0])
            direction = direction / np.linalg.norm(direction)
            projected_point = np.array([p.x, p.y]) + d_required * direction
            envelope_points.append(projected_point)

    return envelope_points  # list of np.array with the green points to draw

def compute_drequired_for_trajectory(v_ego, acc_confort, priority, t_reaction, v_other): 
    """Calculates the minimum distance with respect to a certain acceleration and reaction time"""
    if priority: 
        return 0
    else:
        return v_other * (t_reaction + v_ego / acc_confort)
    
def compute_visibility_polygon(ego_pos, angles, visibility):
    """Builds the real visibility polygon from ray casting"""
    points = [ego_pos + visibility[i] * np.array([np.cos(angles[i]), np.sin(angles[i])]) 
              for i in range(len(angles))]
    return Polygon(points)

def create_roundabout(center=(0,0), radius=15, num_points=30):
    """Returns a polygon approximating a roundabout circle"""
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    return [(center[0] + radius*np.cos(a), center[1] + radius*np.sin(a)) for a in angles]
