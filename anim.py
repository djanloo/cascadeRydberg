from fryd.ercolation import shells_by_cells, set_seed
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from rich.progress import track
from matplotlib import cm

r = 0.02
d = 0.005

core_area = np.pi*(r-d/2)**2*100000
ball_area = np.pi*(r+d/2)**2*100000

decay_p = 0.01
excitation_p = 0.1

N_points = 80*80
N_iter = 2

np.random.seed(43)
set_seed(42)

fig, ax = plt.subplots()
ax.axis("off")
ax.set_aspect("equal")
ax.set_title(f"$p_e$ = {excitation_p} $p_d$ = {decay_p} N={N_points}")

S = np.random.uniform(0,1, size=(N_points, 2)).astype(np.float32)
# x,y = np.meshgrid(np.linspace(0,1, 80), np.linspace(0,1,80))
# S = np.stack((x.flatten(), y.flatten()), axis=1).astype(np.float32)

cmap = cm.get_cmap('plasma', 4)

scat = ax.scatter(S[:,0], S[:,1], c=np.zeros(N_points), vmin=0, vmax=3, cmap=cmap)
colorbar = fig.colorbar(scat)
colorbar.set_ticks([0.5, 1.25, 2, 2.75], labels=["EXTERNAL", "SHELL", "INTERNAL", "CORE"])
results = shells_by_cells(S, r, d, 
                            excitation_probability=excitation_p, decay_probability=decay_p,
                            N_iterations=N_iter
                            )
def update(i):
    print(i)
    global results
    results = shells_by_cells(S, r, d, 
                            excitation_probability=excitation_p, decay_probability=decay_p,
                            N_iterations=N_iter, initial_cores=results["cores"]
                            )
    scat.set_array(results["topological_state"])


anim = FuncAnimation(fig, update, interval=1000/60, frames=300)
anim.save("random_lattice.mp4")
plt.show()