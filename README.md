## CRyd Ercolation

Numerical simulation of the avalanche process of Rydberg atom exitation through percolation.

The aim is to simulate ~10^6 (steady) atoms. The simulation requires to compute repeatedly a fixed-radius near neighbor list.
Due to the high number of particles a "distances look-up table" strategy cannot be pursued since the matrix itself would occupy 10^12 * 4 bytes = 4 TB of memory.

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

![fig1](Figure_1.png)
![fig4](Figure_4.png)



