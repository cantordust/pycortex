# Overview

Python-based interface to PyTorch for evolving deep learning models.

For now, the library supports the following operations:

- Add / erase layers (convolutional or fully connected)
- Add / erase nodes (or kernels in convolutional layers)
- Grow or shrink strides
- Grow or shrink kernels
- Crossover / cloning deep NNs

All mutation and crossover operations should produce fully functional models which can be further trained with backprop.

All functionality will be backported to the C++ version of Cortex (with Python bindings for the die-hard Python fans).

# Dependencies

Install PyTorch if you haven't done so already. PyCortex runs only on CPUs for now (GPU support is WIP), so please install the CPU-only version for better performance. The GPU-enabled version will work, but it may not be optimal (this needs to be confirmed). Installation instructions can be found at (the PyTorch website)[https://pytorch.org/].

# Installation

To setup a dev version of the library:

```
$> git clone https://gitlab.com/cantordust/pycortex
$> cd pycortex
$> pip3 install --user -e .
```
PyCortex works only with Python 3.

This should install all the dependencies. If any of them are already installed and you want to keep an old version for some reason, just remove the entry for that dependency from `setup.py`.

# Run unit tests
Note: Some unit tests are not working at the moment due to significant changes in the backend. I haven't had time to fix them as I have been working on the core, but I will update this as soon as they are all working again.

# Run the MNIST example
```
$> cd experiments/mnist
$> mpirun --map-by core --np <number-of-cores> python3 mnist.py
```
