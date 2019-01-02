# Overview

Python-based interface to PyTorch for evolving deep learning models.

For now, the library supports the following operations:

- Add / erase layers (convolutional or fully connected)
- Add / erase nodes (or kernels in convolutional layers)
- Grow or shrink kernels
- Crossover / cloning deep NNs

All mutation and crossover operations should produce fully functional models which can be further trained with backprop.

All functionality will be backported to the C++ version of Cortex (with Python bindings for the die-hard Python fans).

# Installation

To setup a dev version of the library:

```
$> python3 -m venv PyCortex
$> cd PyCortex && . ./bin/activate
$> pip3 install --user -e .
```

# Run unit tests

```
$> cd unit_tests
$> python3 <unit_test_script>
```

# Run the MNIST example

```
$> cd experiments/MNIST
$> python3 mnist.py
```
