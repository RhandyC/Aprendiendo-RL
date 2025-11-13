import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from common_utils import Simulation

# Run simulation
print("Starting perception envelope evolution simulation...")
sim = Simulation("empty")
bus = [
        (20.0, -1/2), 
        (32.0, -1/2),
        (32.0, -1/2 - 4.0),
        (20.0, -1/2 - 4.0), 
    ]

new_trajectory = [
            (0.0, 0.0),
            (5.0, 0.0),
            (10.0, 0.0),
            (15.0, 0.0),
            (20.0, 0.0),
            (25.0, 0.0)
        ]
initial_point = (50,20)
final_point = (20,30)
centerlane = [initial_point, final_point]

sim.add_obstacle(bus)
sim.set_trajectory(new_trajectory)
sim.add_centerlane(centerlane)


# Create animation
animation = FuncAnimation(sim.fig, sim.update, frames=600, interval=1, blit=False)

plt.show()

# At the end, show evolution graphs
# plot_visibility_evolution(sim)