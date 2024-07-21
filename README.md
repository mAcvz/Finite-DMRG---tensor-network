# Tensor Network Project
This project implements various functionalities for tensor networks, focusing on Matrix Product States (MPS) and Density Matrix Renormalization Group (DMRG) algorithms. The implementation is structured into several Python scripts, each handling different aspects of the computations.

## Project Background

The original objective of this project was to study a fluxonium quantum circuit, a superconducting qubit system used in quantum computing research. However, the scope of the project evolved, and the primary focus became the implementation of the Density Matrix Renormalization Group (DMRG) algorithm within the tensor network formalism. This implementation serves as a foundational tool for future studies on quantum circuits and other one-dimensional quantum systems.

## Libraries:
- "functions_mps.py" contains the basical functions needed to work in the MPS formalism
- "functions_dmrg.py" contains the DMRG functions (Lanczos and sweeps)
- "total.py" contains the structure of the DMRG algorithm, calling all the other functions 

### Run files:
- "long_run.py" calls the DMRG function using different parameters in order to probe different parameters

### Graphical tools:
- "plots.html" is the code we used to create the graphical results shown in the presentation

Documents:
- "presentation.pdf" is the final presentation (of both the theory and the results part)
- "task.pdf" is the assignment document

## Overview

### Matrix Product States (MPS)

MPS represents the quantum state as a product of tensors, which allows for efficient computation of various properties.

### Density Matrix Renormalization Group (DMRG)

DMRG is a powerful method for finding the ground state of one-dimensional quantum systems. It works by iteratively optimizing the MPS representation of the quantum state, targeting the minimum energy configuration.
