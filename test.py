from fryd.regular_solid import regular_lattice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 50
S = np.zeros((N, N), dtype=np.uintc)
stats = regular_lattice(S, N_iterations=1)

fig, ax = plt.subplots()
img = ax.matshow(stats["state"])
# plt.show()
def update(i):
    global stats
    stats = regular_lattice(S, N_iterations=1, startfrom=stats["state"], 
                            excitation_probability=0.1, decay_probability=0.24)
    if i < 100:
        img.set_data(stats["state"])

anim = FuncAnimation(fig, update, interval=1, frames=100)

plt.show()