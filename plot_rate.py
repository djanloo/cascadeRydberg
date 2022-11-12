from fryd.ercolation import shells_by_cells
import numpy as np
from matplotlib import pyplot as plt
from rich.progress import track

space_dimension = 2
r = 0.05
d = 0.008
decay_p = 0.3
excitation_p = 1e-1

N_points = 2000
N_iter = 1000

print(f"mean_excitated_points_in_ball = {excitation_p*np.pi*N_points*( (r + d/2)**space_dimension )  }")

samples = 1
M = 2

n = np.zeros((samples*10, M))
sigma = np.zeros((samples*10,M))
rates = np.linspace(1e-3, 0.9, M)

S = np.random.uniform(0,1, (N_points, space_dimension)).astype(np.float32)

_ , diagnostics = plt.subplots()
_, configuration = plt.subplots()
configuration.set_aspect("equal")

fig_vera, ax = plt.subplots()
for samp in range(samples):
    for j, p in track(enumerate(rates), total=M):
        results = shells_by_cells(S, r, d, 
                                excitation_probability=p, decay_probability=decay_p,
                                N_iterations=N_iter
                                )
        n[samp*10:(samp+1)*10, j]      = results["N_cores_t"][-100:-1:10]/N_points
        sigma[samp*10:(samp+1)*10, j ] = results["N_cores_t"][-100:-1:10]/N_points

        diagnostics.plot(results["N_cores_t"]/N_points)

upper, median, lower = np.quantile(n, [.2, .5, .8], axis=0)

configuration.scatter(results["state"][:,0 ], results["state"][:,1], c=results["topological_state"], vmin=0, vmax=3)
ax.set_title(f"Decay probability = {decay_p:.3}", size=10)
ax.set_xlabel("excitation probability", size=10)
ax.set_ylabel("$\psi$", size=10)

ax.fill_between(rates, lower, upper, color='orange', alpha=0.5)
ax.plot(rates, median, color='k')
fig_vera.savefig("rate_criticality.pdf")
plt.show()