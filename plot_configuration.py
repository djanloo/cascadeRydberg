from fryd.ercolation import shells_by_cells, set_seed
import numpy as np
from matplotlib import pyplot as plt
from rich.progress import track

r = 0.02
d = 0.005
decay_p = 0.2
excitation_p = 0.3

N_points = 6000
N_iter = 1000
np.random.seed(43)
set_seed(42)
S = np.random.uniform(0,1, size=(N_points, 2)).astype(np.float32)

results = shells_by_cells(S, r, d, 
                        excitation_probability=excitation_p, decay_probability=decay_p,
                        N_iterations=N_iter
                        )
fig, ax = plt.subplots()
ax.set_aspect('equal')
top_state = np.array(results["topological_state"])
ax.set_title(f"$r = {r}, delta = {d}, p_e = {excitation_p}, N = {N_points}, iter = {N_iter}$", size=10)
plt.scatter(S[:,0], S[:,1], c=top_state, vmin=0, vmax=3)
plt.show()