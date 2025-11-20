import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from common_utils import Simulation

# Run simulation
print("Starting perception envelope evolution simulation...")
sim = Simulation("roundbout")

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
    animation.save("custom_uscase.mp4", writer=writer)