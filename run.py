from cryd.ercolation import run
import numpy as np
from rich.progress import track

for i in track(range(10)):
    S = np.random.uniform(0,1, size=(100,3))
    NN = run(S, 0.5)
    print(len(NN))