## CRyd Ercolation

Numerical simulation of the avalanche process of Rydberg atom exitation through percolation.

![fig1](shells_by_cells_2.png)
![fig4](rate_plot.png)

### Requirements and installation
`pipenv` is required.

To activate the environment run

```
pipenv install
pipenv shell
```

then in the environment run

```
make
```

to compile the cython code.

Other compilation methods: 
- `make profile` for profiling
- `make hardcore` for speedup
- `make hardcoreprofile` for profiling and speedup



