from fryd.ercolation import shells_by_cells, set_seed
import numpy as np
from matplotlib import pyplot as plt
from rich.progress import track

r = 0.02
d = 0.005
decay_p = 0.03
excitation_p = 0.3

N_points = 6000
N_iter = 3
np.random.seed(43)
set_seed(42)
S = np.random.uniform(0,1, size=(N_points, 2)).astype(np.float32)

results = shells_by_cells(S, r, d, 
                        excitation_probability=excitation_p, decay_probability=decay_p,
                        N_iterations=N_iter
                        )
fig, ax = plt.subplots(2)
fig.suptitle(f"$r = {r}, delta = {d}, p_e = {excitation_p}, p_d = {decay_p}, N = {N_points}, iter = {N_iter}$", size=10)
ax[0].set_title("Last configuration", size=10)
ax[0].set_aspect('equal')
ax[0].axis("off")
top_state = np.array(results["topological_state"])
ax[0].scatter(S[:,0], S[:,1], c=top_state, s=2, vmin=0, vmax=3)

ax[1].plot(results["N_cores_t"]/N_points, label="N_cores")
ax[1].plot(results["N_decayed_t"]/N_points, label="N_decayed")
ax[1].plot(results["N_inflated_t"]/N_points, label="N_inflated")
ax[1].set_title("Time analysis", size=10)
ax[1].legend(fontsize=10)
ax[1].set_xlabel("iteration", size=10)
ax[1].grid(ls=":")
plt.show()