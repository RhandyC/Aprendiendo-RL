import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from common_utils import Simulation

# Run simulation
print("Starting perception envelope evolution simulation...")
sim = Simulation("t_joint")

# Create animation
animation = FuncAnimation(sim.fig, sim.update, frames=600, interval=1, blit=False)

plt.show()