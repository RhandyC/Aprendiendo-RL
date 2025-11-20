import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from common_utils import Simulation, Centerlane

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
            (-20.0, 0.0),
            (10.0, 0.0),
            (20.0, 0.0),
            (30.0, 0.0),
            (40.0, 0.0),
            (100.0, 0.0)
        ]

initial_point = (50,-50)
final_point = (20,50)
centerlane1 = Centerlane(initial_point, final_point, 3.0)

initial_point = (8,50)
final_point =(15,-50)
centerlane2 = Centerlane(initial_point, final_point, 10.0)

initial_point = (80,50)
final_point =(80,-50)
centerlane3 = Centerlane(initial_point, final_point, 5.0)

# sim.add_obstacle(bus)
sim.set_trajectory(new_trajectory)
sim.add_centerlane(centerlane1)
sim.add_centerlane(centerlane2)
sim.add_centerlane(centerlane3)

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
    animation.save("custom_usecase.mp4", writer=writer)
