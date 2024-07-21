# Tensor Network Project

This project implements various functionalities for tensor networks, focusing on Matrix Product States (MPS) and Density Matrix Renormalization Group (DMRG) algorithms. The implementation is structured into several Python scripts, each handling different aspects of the computations.

## Project Structure

### Files

- **long_run.py**: This script is likely the main entry point of the project, coordinating the execution of tensor network operations using the functions defined in the other scripts.
- **functions_mps.py**: Contains functions and utilities related to Matrix Product States (MPS), a key component of tensor networks used to efficiently represent quantum states.
- **functions_dmrg.py**: Includes functions and algorithms for the Density Matrix Renormalization Group (DMRG), a numerical variational technique designed to find the low-energy states of quantum many-body systems.

## Overview

### Matrix Product States (MPS)

MPS represents the quantum state as a product of tensors, which allows for efficient computation of various properties.

### Density Matrix Renormalization Group (DMRG)

DMRG is a powerful method for finding the ground state of one-dimensional quantum systems. It works by iteratively optimizing the MPS representation of the quantum state, targeting the minimum energy configuration.
