# qarray-rust-core

![GitHub Workflow Status](https://github.com/b-vanstraaten/qarray-rust-core/workflows/CI/badge.svg)
![PyPI](https://img.shields.io/pypi/v/rusty-capacitance-model-core)

![Rust logo](https://www.rust-lang.org/static/images/rust-logo-blk.svg)
![Python logo](https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/files/python-logo-only.svg)

**Quantum Dot Constant Capacitance Simulator** is a high-performance Python package that leverages the power of Rust and Rayon to provide a fully parallelised and optimised simulation environment for quantum dots with constant capacitance.

**This package provides core functionality; it is not intended that the user will interact with it directly.**

## Features

- **Ultra-fast Simulation:** Harnesses the speed of Rust and the parallelism of Rayon to deliver lightning-fast simulations.
- **Constant Capacitance:** Specialized for simulating quantum dots with constant capacitance, allowing precise modelling of charge dynamics.
- **User-Friendly:** Designed with ease of use in mind, making it accessible to both experts and newcomers in quantum dot simulations.
- **Extensive Documentation:** Comprehensive documentation and examples to help you get started quickly.

## Installation

Install Quantum Dot Constant Capacitance Simulator using pip:

```bash
pip install qarray-rust-core
```


### Usage

This package exposes two functions to be called from python: 

- `ground_state_open` - computes the lowest energy state of a quantum dot array with constant capacitance and which is open, such that the total number of changes is not fixed. 
- `ground_state_closed` - computes the lowest energy state of a quantum dot array with constant capacitance and which is closed, such that the total number of changes is fixed.

The python code to call these functions is as follows:

   ```python
   from qarray-rust-core import (ground_state_open, ground_state_closed)
   import numpy as np 
   
   # the dot-dot capacitance matrix
   cdd = np.array([
        [1, -0.1],
        [-0.1, 1]
   ])
   cdd_inv = np.linalg.inv(cdd)
   
   # the dot-gate capacitance matrix
   cgd = np.array([
          [1, 0.3],
          [0.3, 1]
    ])
   
   # define a matrix of gate voltages to sweep over the first gate
   vg = np.stack([np.linspace(-1, 1, 100), np.zeros(100)], axis = -1)
   
   n_charge = 3 # the number of changes to confine in the quantum dot array for the closed case 
   threshold = 1 # threshold to avoid having to consider all possible charge states, setting it 1 is always correct, however has a computatinal cost. 
   
   n_open = ground_state_open(vg, cgd, cdd_inv, threshold)
   n_closed = ground_state_closed(vg, n_charge, cgd, cdd, cdd_inv, threshold)
   ```
**It is not intended the user ever call these functions directly.**

There is a pure Python wrapper that provides a more user-friendly interface to this core functionality. 
See [Quantum Dot Constant Capacitance Simulator](https://github.com/b-vanstraaten/rusty_capacitance_model). This package provides: 

- **A user-friendly interface** to the core functionality.
- **Plotting, charge sensing, virtual gate** and gate voltage sweeping (1d and 2d) functionality.
- **Advanced type checking** using pydantic.
- **Automated testing** including for the functionality in this package.
- **More examples**.