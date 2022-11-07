from cryd.ercolation import run
import numpy as np
from matplotlib import pyplot as plt
from rich.progress import track

M = 50

S = np.random.uniform(0,1 , size=(500, 3)).astype(np.float32)
n = np.zeros(M)
eps = np.linspace(0,0.2,M)

for _ in range(5):
    for i, r in track(enumerate(eps), total=M):
        n[i] = run(S, r)

    plt.plot(eps, n)
plt.show()