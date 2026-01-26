import json
import re

# Notebook Structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        },
        "colab": {
            "provenance": [],
            "gpuType": "A100"
        },
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

def add_cell(source_code, cell_type="code"):
    if not source_code.strip(): return
    lines = [line + "\n" for line in source_code.splitlines()]
    if lines: lines[-1] = lines[-1].rstrip("\n")
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    notebook["cells"].append(cell)

# 1. Header Cell
add_cell("# Stochastic Higher-Order Swendsen-Wang vs WalkSAT\n\nThis notebook compares our **Stochastic Cluster Monte Carlo** algorithm against the industry standard for Random SAT: **WalkSAT**.\n\n## The Contenders\n1.  **Stochastic Swendsen-Wang (Ours)**:\n    *   Physics-based (Cluster Dynamics).\n    *   Uses geometric frustration and percolation.\n    *   **New**: Uses **Exact Hamiltonian Cluster Updates** (Exact Energy Delta) for decision.\n    *   **Schedule**: Logarithmic annealing (dense near $\\omega_{max}$).\n    *   Runs on GPU (Massively Parallel).\n2.  **WalkSAT (Reference)**:\n    *   Stochastic Local Search.\n    *   Greedy + Noise heuristic.\n    *   Runs on CPU (Sequential, fast flips).\n3.  **Dynamics UNSAT (New)**:\n    *   Focuses dynamics on clusters touching UNSAT clauses (Rejection Sampling).\n", cell_type="markdown")

# 2. Read and Parse Python Script
with open("simulate_dynamics.py", "r", encoding="utf-8") as f:
    content = f.read()

# Split by markers
# Markers in file:
# # --- KERNELS ---
# # --- GENERATOR ---
# # --- SOLVERS ---
# # --- MAIN ---

parts = re.split(r'# --- [A-Z]+ ---', content)

# parts[0]: Imports and Setup
# parts[1]: Kernels
# parts[2]: Generator
# parts[3]: Solvers
# parts[4]: Main Loop

titles = [
    "# @title 1. Environment & GPU Setup",
    "# @title 3. Shared Kernels",
    "# @title 2. Data Generators",
    "# @title 3. Solvers (Including DynamicsUNSAT_GPU)",
    "# @title 5. Main Simulation Loop"
]

for i, part in enumerate(parts):
    title = titles[i] if i < len(titles) else ""
    code = part.strip()
    if code:
        full_cell_content = f"{title}\n{code}"
        add_cell(full_cell_content, cell_type="code")

# 3. Save Notebook
with open("Swendsen-Wang_3SAT_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook 'Swendsen-Wang_3SAT_Colab.ipynb' successfully regenerated from 'simulate_dynamics.py'.")
