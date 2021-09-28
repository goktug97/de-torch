Minimal PyTorch Library for Differential Evolution
========================================================

# Requirements

```
numpy
torch
```

## Optional
```
mpi4py # If you want to run in parallel
gym # For examples
```

To install `mpi4py` you need `MPI` installed in the system.
Check: https://mpi4py.readthedocs.io/en/stable/install.html

A Dockerfile is provided for convenience.

# Installation

```bash
pip install detorch --user
```

## Source
```bash
git clone --recursive https://github.com/goktug97/de-torch
cd de-torch
python3 setup.py install --user
```

# Usage
See https://github.com/goktug97/de-torch/blob/master/examples

Check https://github.com/goktug97/de-torch/blob/master/detorch/config.py for parameters.

You can run the example with
```bash
PYTHONPATH="$(pwd):$PYTHONPATH" python examples/example.py
```
or in parallel for faster training (requires mpi4py).
```bash
PYTHONPATH="$(pwd):$PYTHONPATH" mpirun -np 2 python examples/example.py
```

# Another Evolution Library for PyTorch

https://github.com/goktug97/nes-torch
