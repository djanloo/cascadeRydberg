from fryd.random_solid import shells_by_cells, set_seed
import numpy as np
from matplotlib import pyplot as plt
from rich.progress import track
from scipy.signal import correlate
r = 0.08
d = 0.01

decay_p = 1e-2
excitation_p = 1e-2

N_points = 600

print(f"mean_excitated_points_in_ball = {excitation_p*np.pi*N_points*( (r + d/2)**2  )  }")

N_iter = 400
Nsamples = 5
np.random.seed(41)
set_seed(42)
fig, ax = plt.subplots(2)
fig.suptitle(f"$r = {r}, delta = {d}, p_e = {excitation_p}, p_d = {decay_p}, N = {N_points}, iter = {N_iter}$", size=10)

def autocorr(y):
    x = y.copy()
    x -= np.mean(x)
    x /= np.std(x)
    result = correlate(x, x, mode='full')/len(x)
    return result[result.size//2:]



timeseries = np.zeros((Nsamples,N_iter-1))
autocorrelations = np.zeros((Nsamples,N_iter-1 - 100))
alive = np.ones(Nsamples).astype(bool)

# Thermalization
S = np.random.uniform(0,1, size=(N_points, 2)).astype(np.float32)
results = shells_by_cells(S, r, d, 
                            excitation_probability=excitation_p, decay_probability=decay_p,
                            N_iterations=N_iter
                            )
for i in track(range(Nsamples)):
    # S = np.random.uniform(0,1, size=(N_points, 2)).astype(np.float32)

    results = shells_by_cells(S, r, d, 
                            excitation_probability=excitation_p, decay_probability=decay_p,
                            N_iterations=N_iter #, initial_cores=results["cores"]
                            )
    alive[i] = not results["died"] 
    timeseries[i] = results["N_cores_t"][:-1]/N_points
    autocorrelations[i] = autocorr(timeseries[i,100:])
    ax[0].plot(timeseries[i])
    ax[1].plot(autocorrelations[i])
    ax[0].plot(results["N_inflated_t"]/N_points, alpha=0.6)

ax[0].set_title("Time analysis of $\psi$", size=10)
# ax.legend(fontsize=10)
ax[0].set_xlabel("iteration", size=10)
ax[0].grid(ls=":")

ax[0].plot(np.mean(timeseries[alive], axis=0), color='k')

ax[1].set_title("Autocorrelation", size=10)
# ax.legend(fontsize=10)
ax[1].set_xlabel("iteration (post thermalization )", size=10)
ax[1].grid(ls=":")
ax[1].plot(np.mean(autocorrelations[alive], axis=0), color='k')



plt.show()