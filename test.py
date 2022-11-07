dim = 3

for u in range(3**dim):
    deltas = [5]*dim
    for k in range(dim):
     deltas[k] = (u//(3**k))%3 - 1 
