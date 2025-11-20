import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from common_utils import Simulation

# Run simulation
print("Starting perception envelope evolution simulation...")
sim = Simulation("angle_joint")

# sim.plot_frame(10)

ANIMATION = False
RECORD = True

if ANIMATION: 
    # Create animation
    animation = FuncAnimation(sim.fig, sim.update, frames=601, interval=1, blit=False)
    plt.show()

if RECORD: 
    animation = FuncAnimation(sim.fig, sim.update, frames=601, interval=1, blit=False)
    # Crear un escritor (opcional: puedes ajustar fps, bitrate, etc.)
    writer = FFMpegWriter(fps=30, bitrate=1800)

    # Guardar la animación
    animation.save("angle_joint.mp4", writer=writer)