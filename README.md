# SW3-SAT: Higher-Order Monte Carlo Cluster Dynamics

This repository contains an optimized GPU implementation of Swendsen-Wang Cluster Dynamics adapted for 3-SAT problems, inspired by recent advances in statistical physics for community detection.

## Context

Based on the methodologies described in:
1.  **SODA 2026**: *A higher-order Monte Carlo cluster dynamics for community detection in spatially embedded graphs*.
2.  **Asilomar 2025**: *Higher-order Monte Carlo cluster dynamics for community detection in graphs*.

The core idea is to alleviate **frustration** in the energy landscape by lifting 3-body interactions (clauses) into higher-order structures (Tetrahedrons) and performing non-local cluster updates.

## Implementation Details

The solver (`TetraDynamicsGPU`) is implemented in Python using **CuPy** for CUDA acceleration. It features:

*   **Mapping:** 3-SAT clauses are mapped to Tetrahedrons using a ghost/dummy node technique to fit the 4-body interaction model.
*   **Vectorized Bond Sampling:** Implements a rigorous 3-state bond percolation ($B=0,1,2$) derived from the energy levels of an isotropic tetrahedron ($0, \omega, 4\omega$).
*   **Ghost Node Clustering:** Handles spin freezing via auxiliary "Ghost" nodes in the graph connectivity, allowing for efficient standard Union-Find operations on the GPU.
*   **Random Priority Witnesses:** A novel, loop-free mechanism to select "witness" variables for freezing bonds without inducing bias.

## Usage

Open `sw3sat_colab.ipynb` in Google Colab (ensure a GPU runtime is selected).

```python
# Key Hyperparameters
omega = 3.5  # Interaction strength
N = 5000     # Number of variables
alpha = 4.2  # Clause density (near phase transition)
```
