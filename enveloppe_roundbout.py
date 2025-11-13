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
    """Builds the real visibility polygon from ray casting"""
    points = [ego_pos + visibility[i] * np.array([np.cos(angles[i]), np.sin(angles[i])]) 
              for i in range(len(angles))]
    return Polygon(points)

def create_roundabout(center=(0,0), radius=15, num_points=30):
    """Returns a polygon approximating a roundabout circle"""
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    return [(center[0] + radius*np.cos(a), center[1] + radius*np.sin(a)) for a in angles]

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


def compute_visibility_coefficient(real_env, required_env, centerlane):
    """Calculates the visibility coefficient"""

    real_st = [global_to_st(p, centerlane) for p in real_env]
    required_st = [global_to_st(p, centerlane) for p in required_env]

    # Extract only the longitudinal s-axis
    real_s = [s for s, _ in real_st]
    required_s = [s for s, _ in required_st]

    # If there are no points in real_s or required_s, there is no intersection
    if len(real_s) == 0 or len(required_s) == 0:
        return 0.0, (None, None)

    # Calculate overlapping intervals in s
    sA_min, sA_max = np.min(required_s), np.max(required_s)
    sB_min, sB_max = np.min(real_s), np.max(real_s)

    inter_min = max(sA_min, sB_min)
    inter_max = min(sA_max, sB_max)

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


def rect_to_poly(rect):
    x, y, w, h = rect
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

def compute_drequired_for_trajectory(v_ego, acc_confort, priority, t_reaction, v_other): 
    """Calculates the minimum distance with respect to a certain acceleration and reaction time"""
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
                intersection_dist, point = self.ray_centerlane_intersection(ego_pos, ray_end, centerlane)
                if intersection_dist is not None and intersection_dist < min_distance:
                    # min_distance = intersection_dist
                    intersection_point = point 
                    visibility_points.append(intersection_point)
            visibility_map[i] = min_distance

        return angles, visibility_map, visibility_points
    
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

class AutonomousVehicle:
    def __init__(self, initial_pos=np.array([0.0, 0.0])):
        self.position = initial_pos
        self.current_target_idx = 0
        self.speed = 3.0  # m/s
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

class Simulation:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.vehicle = AutonomousVehicle()
        
        # Define bus as a bounding box [x, y, width, height]
        self.bus = np.array([40.0, -2.0, 12.0, 4.0])
        self.corner1 =  np.array([-50.0, -55.0, 100.0, 50.0]) # Position and dimensions
        self.corner2 =  np.array([-50.0, 10.0, 100.0, 50.0]) # Position and dimensions

        # Obstacles defined as 4-edge polygones 
        self.corner1_poly = rect_to_poly([-50.0, -55.0, 100.0, 50.0])
        self.corner2_poly = rect_to_poly([-50.0, 10.0, 100.0, 50.0])

        self.roundabout_poly = create_roundabout(center=(50, 0), radius=15)

        # New irregular obstacle
        angle = np.pi/3
        ego_lane_width = 7.5

        # np.array([[initial_point],   # initial point
        #                       [final_point]])  # final point

        # Other obstacles (buildings, other vehicles)
        self.obstacles = [
            self.roundabout_poly,
        ]

        self.centerlanes = [
        ]
        
        self.vehicle.trajectory = [
            (0, 0),
            (20, 0),
            (22, 0),
            (25, -10),
            (30, -20),
            (40, -10),   # Centro de la rotonda
            (50, 0),   # Salida
            (80, 0)
        ]

        self.frame_count = 0
        self.visibility_data = []
        
    def update_weather_conditions(self):
        """Simulates changes in weather conditions and resets upon change"""
        conditions = ["clear", "rain", "fog", "heavy_fog"]
        condition_idx = (self.frame_count // 150) % len(conditions)
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
        # required_env_array = compute_required_envelope(self.vehicle.position)
        # required_env = Polygon(required_env_array)  # now it's a Polygon

        # # Draw the envelope as a semi-transparent blue polygon
        # self.ax.fill(
        #     required_env_array[:, 0], required_env_array[:, 1],
        #     color='blue', alpha=0.2, label='Required envelope'
        # )
        text_lines = []

        for idx, centerlane in enumerate(self.centerlanes):
            required_env = compute_required_envelope(
                self.vehicle.position,
                self.vehicle.trajectory,
                centerlane,
                v_ego=self.vehicle.speed,
                acc_confort=2.5,
                t_reaction=0.5,
                v_other=50.0/3.6,
                priority=False
            )
            
            # Draw green points projected onto the lanes
            for p in required_env:
                self.ax.plot(p[0], p[1], 'go', markersize=5)
            
            # Calculate visibility coefficient
            try:
                C_v, (inter_init_point, inter_last_point) = compute_visibility_coefficient(
                    intersection_points, required_env, centerlane
                )
            except TopologicalError:
                C_v = 0.0
                inter_init_point = inter_last_point = None

            # Draw intersection segment if it exists
            if inter_init_point is not None and inter_last_point is not None:
                self.ax.plot(
                    [inter_init_point[0], inter_last_point[0]],
                    [inter_init_point[1], inter_last_point[1]],
                    'b-o', markersize=2, linewidth=2, label=f'Intersection lane {idx+1}'
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
        
        return []
    
    def plot_scenario(self, angles, visibility, visible_ratio, avg_distance, intersection_points):
        """Visualizes the complete scenario"""
        # Configure the plot
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Draw obstacles
        for obstacle_vertices in self.obstacles:
            # x, y, w, h = obstacle
            # rect = patches.Rectangle((x, y), w, h, linewidth=2, 
            #                        edgecolor='red', facecolor='darkred', alpha=0.7)
            # self.ax.add_patch(rect)
            poly_patch = patches.Polygon(obstacle_vertices, closed=True, linewidth=2,
                                 edgecolor='red', facecolor='darkred', alpha=0.7)
            self.ax.add_patch(poly_patch)
        
        # Draw centerlanes
        for idx, centerlane in enumerate(self.centerlanes):
            init_point, final_point = centerlane
            dx = final_point[0] - init_point[0]
            dy = final_point[1] - init_point[1]

            # Draw the arrow
            self.ax.arrow(
                init_point[0], init_point[1],
                dx, dy,
                head_width=2, head_length=8,
                fc='blue', ec='blue'
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

        # # Calculate required points according to the formula
        # d_required_points = compute_required_envelope(self.vehicle.position,
        #                                             self.vehicle.trajectory,
        #                                             self.centerlanes,
        #                                             v_ego=self.vehicle.speed,
        #                                             acc_confort=2.5,
        #                                             t_reaction=0.5,
        #                                             v_other=50.0/3.6,
        #                                             priority=False)
        # # Draw green points projected onto the lanes
        # for p in d_required_points:
        #     self.ax.plot(p[0], p[1], 'go', markersize=5)  # 'go' = green circle

        # Draw autonomous vehicle
        vehicle_circle = patches.Circle(self.vehicle.position, radius=1.5, 
                                      facecolor='blue', edgecolor='darkblue', linewidth=2)
        self.ax.add_patch(vehicle_circle)
        
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
                    f"Visible Area: {visible_ratio*100:.1f}%\n"
                    f"Avg. Visible Dist: {avg_distance:.1f}m")
        
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.ax.set_title(f"Evolution of Perception Envelope - Frame {self.frame_count}")

    def reset_simulation_state(self):
        """Resets the simulation's position and data"""
        self.vehicle.position = np.array([0.0, 0.0])
        self.visibility_data.clear()
        print(f"--- Resetting simulation due to weather change: {self.vehicle.perception.weather_condition} ---")


def plot_visibility_evolution(simulation):
    """Graphs the evolution of visibility over time"""
    if len(simulation.visibility_data) == 0:
        return
    
    distances = [data[0] for data in simulation.visibility_data]
    visible_ratios = [data[1] * 100 for data in simulation.visibility_data]  # Percentage
    avg_distances = [data[2] for data in simulation.visibility_data]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Visible area vs distance graph
    ax1.plot(distances, visible_ratios, 'b-', linewidth=2, label='Visible Area (%)')
    ax1.set_xlabel('Distance to Bus (m)')
    ax1.set_ylabel('Percentage of Visible Area (%)')
    ax1.set_title('Evolution of Visible Area vs Distance to Bus')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.invert_xaxis()  # Because the distance decreases
    
    # Average visible distance graph
    ax2.plot(distances, avg_distances, 'g-', linewidth=2, label='Average Visible Distance')
    ax2.axhline(y=simulation.vehicle.perception.sensor_range, color='r', linestyle='--', 
                label='Maximum Sensor Range')
    ax2.set_xlabel('Distance to Bus (m)')
    ax2.set_ylabel('Visible Distance (m)')
    ax2.set_title('Evolution of Average Visible Distance')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.invert_xaxis()
    
    plt.tight_layout()
    plt.show()

# Run simulation
print("Starting perception envelope evolution simulation...")
sim = Simulation()

# Create animation
animation = FuncAnimation(sim.fig, sim.update, frames=600, interval=1, blit=False)

plt.show()

# At the end, show evolution graphs
plot_visibility_evolution(sim)