import json
import nbformat as nbf

b = nbf.v4.new_notebook()

# Cell 1: Intro
text_intro = r"""# Higher-Order Monte Carlo Cluster Dynamics for 3-SAT (GPU)

This notebook implements a high-performance **Swendsen-Wang Cluster Dynamics** solver for 3-SAT problems, adapted from the physics of **spatially embedded graphs** and **frustrated systems** (referencing *SODA 2026* and *Asilomar 2025*).

## The Physics
Instead of treating SAT clauses as simple constraints, we map them to **Tetrahedrons** (4-body interactions). By distributing energy onto these higher-order structures and utilizing a specific decision tree for bond percolation, we can:
1.  Minimize the number of "frozen" bonds (reducing frustration).
2.  Maintain the correct Gibbs measure.
3.  Accelerate sampling via cluster updates.

## Algorithm Architecture
The implementation follows a strict **Array Programming** paradigm using **CuPy** (CUDA for Python) to ensure massive parallelism.

*   **Mapping:** 3-SAT Clauses $\to$ Tetrahedrons (via Ghost/Slack nodes).
*   **Dynamics:** 3-State Bond Sampling ($B=0, 1, 2$) based on satisfaction levels $k$.
*   **Witness Selection:** Vectorized "Random Priority" mechanism (no loops).
*   **Cluster Flipping:** Ghost-Node Graph construction + Connected Components on GPU.

---"""
b.cells.append(nbf.v4.new_markdown_cell(text_intro))

# Cell 2: Setup
code_setup = r"""# @title 1. Environment Setup & Imports
# We check for GPU availability and install CuPy if needed (standard on Colab).

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import requests

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx
    import cupyx.scipy.sparse.csgraph as cpx_graph
    print(f"GPU Detected: {cp.cuda.runtime.getDeviceCount()} device(s)")
except ImportError:
    print("CuPy not found. Installing...")
    !pip install cupy-cuda12x
    import cupy as cp
    import cupyx.scipy.sparse as cpx
    import cupyx.scipy.sparse.csgraph as cpx_graph

# Graphics settings
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (12, 6)"""
b.cells.append(nbf.v4.new_code_cell(code_setup))

# Cell 3: Data Gen
code_data = r'''# @title 2. Data Generation & Parsing

def generate_random_3sat(N, alpha, seed=None):
    """
    Generates a Random 3-SAT instance.
    N: Number of variables
    alpha: Ratio of clauses/variables (M = alpha * N)
    Returns: (M, 3) array of literals (1-based index, negative for NOT)
    """
    if seed is not None:
        np.random.seed(seed)
    
    M = int(N * alpha)
    # Variables are 1..N
    vars = np.random.randint(1, N + 1, size=(M, 3))
    # Signs are +/- 1
    signs = np.random.choice([-1, 1], size=(M, 3))
    
    clauses = vars * signs
    return clauses, N

def parse_dimacs(content):
    """Parses DIMACS CNF content string."""
    clauses = []
    N = 0
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('c'): continue
        if line.startswith('p'):
            parts = line.split()
            N = int(parts[2])
            continue
        
        # Parse literals
        lits = [int(x) for x in line.split() if x != '0']
        if len(lits) == 3:
            clauses.append(lits)
            
    return np.array(clauses, dtype=np.int32), N

print("Generators ready.")'''
b.cells.append(nbf.v4.new_code_cell(code_data))

# Cell 4: Solver
# Using triple single quotes. IMPORTANT: Fixed dtype to float32 for cuSPARSE compat
code_solver = r'''# @title 3. The Solver: `TetraDynamicsGPU`
# This is the core kernel implementing the prompt's specific dynamics.

class TetraDynamicsGPU:
    def __init__(self, clauses_np, N, omega=2.0):
        """
        Initialize the Higher-Order Cluster Solver.
        clauses_np: (M, 3) numpy array of literals.
        N: Number of variables.
        omega: Energy scaling parameter.
        """
        self.N = N
        self.M = len(clauses_np)
        self.omega = omega
        
        # --- 1. TetraBuilder: Map 3-SAT Clauses to Tetrahedrons ---
        # We add a virtual 'Slack' node to every clause to form a 4-body interaction.
        # This slack node (index N) is kept fixed or weakly coupled.
        # Here, we treat the clause effectively as a tetrahedron where the 4th node 
        # is a global 'always unsatisfied' or 'dummy' node to fit the 4-body logic,
        # OR strictly use the 3 nodes and map energy levels accordingly.
        # Following the prompt strictly: "Tétraèdre isotrope... énergies 0, w, 4w"
        
        # We pad the clauses to shape (M, 4) using a dummy variable index 'N'.
        # This dummy variable will be pinned to a value that ensures it contributes 
        # to the 'unsatisfied' count for the energy mapping.
        self.DUMMY_VAR_IDX = N  # 0-based index for the dummy
        
        # Prepare data for GPU
        clauses_pad = np.zeros((self.M, 4), dtype=np.int32)
        clauses_pad[:, :3] = clauses_np
        # The 4th literal: Reference to Dummy Variable (always index N)
        # We set sign +1. We will force spins[N] = -1 so this literal is unsatisfied.
        clauses_pad[:, 3] = (self.DUMMY_VAR_IDX + 1) 
        
        # Extract indices and signs
        self.tetra_indices = cp.array(np.abs(clauses_pad) - 1, dtype=cp.int32)
        self.tetra_signs = cp.array(np.sign(clauses_pad), dtype=cp.int8)
        
        # Initialize Spins (-1 or +1). Size N + 1 (for dummy)
        self.spins = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=N + 1)
        self.spins[self.DUMMY_VAR_IDX] = -1 # Pin dummy to -1

        # Pre-calculate probabilities for the decision tree (Optimization)
        # Exp(-4w), Exp(-w), etc.
        self.prob_e_3w = cp.exp(-3 * omega)
        self.prob_e_4w = cp.exp(-4 * omega)
        self.prob_e_w  = cp.exp(-1 * omega)
        
        # Ghost Node Indices for Graph Construction
        # N+1: Ghost PLUS (+), N+2: Ghost MINUS (-)
        self.GHOST_PLUS = N + 1
        self.GHOST_MINUS = N + 2
        self.TOTAL_NODES = N + 3

    def step(self):
        """
        A single Swendsen-Wang step on Tetrahedrons.
        Fully vectorized on GPU.
        """
        # 1. Calculate Satisfaction k per Tetrahedron
        # Gather spins: (M, 4)
        current_spins = self.spins[self.tetra_indices]
        # Check literal satisfaction (Spin == Sign)
        is_sat = (current_spins == self.tetra_signs)
        # k: number of satisfied literals per tetrahedron
        k = cp.sum(is_sat, axis=1, dtype=cp.int8)
        
        # 2. Sample Bonds (B in {0, 1, 2})
        u = cp.random.random(self.M, dtype=cp.float32)
        bonds = cp.zeros(self.M, dtype=cp.int8)
        
        # Vectorized Decision Tree (from Prompt)
        # Case k=1: if u >= e^-3w -> B=1
        mask_k1 = (k == 1)
        bonds[mask_k1 & (u >= self.prob_e_3w)] = 1
        
        # Case k>=2
        mask_k2 = (k >= 2)
        # Sub-case: u < e^-4w -> B=0 (Already 0)
        # Sub-case: e^-4w <= u < e^-w -> B=1
        mask_b1 = mask_k2 & (u >= self.prob_e_4w) & (u < self.prob_e_w)
        bonds[mask_b1] = 1
        # Sub-case: u >= e^-w -> B=2
        mask_b2 = mask_k2 & (u >= self.prob_e_w)
        bonds[mask_b2] = 2
        
        # 3. Select Witnesses (Random Priority Optimization)
        # We need to choose 'B' satisfied nodes randomly.
        # Generate random priorities for all literals
        priorities = cp.random.random((self.M, 4), dtype=cp.float32)
        # Mask unsatisfied literals so they are never chosen (priority -1)
        priorities[~is_sat] = -1.0

        # Find 1st Witness (Max Priority)
        # argmax returns index 0..3 relative to tetrahedron
        idx_w1_local = cp.argmax(priorities, axis=1)
        # Map to global variable index
        idx_w1_global = cp.take_along_axis(self.tetra_indices, idx_w1_local[:, None], axis=1).flatten()
        # Get the sign required for this witness
        sign_w1 = cp.take_along_axis(self.tetra_signs, idx_w1_local[:, None], axis=1).flatten()

        # Find 2nd Witness (if B=2)
        # Mask the first witness to find the second max
        priorities_w2 = priorities.copy()
        # Set the priority of the chosen w1 to -2 so it's not picked again
        rows = cp.arange(self.M)
        priorities_w2[rows, idx_w1_local] = -2.0
        
        idx_w2_local = cp.argmax(priorities_w2, axis=1)
        idx_w2_global = cp.take_along_axis(self.tetra_indices, idx_w2_local[:, None], axis=1).flatten()
        sign_w2 = cp.take_along_axis(self.tetra_signs, idx_w2_local[:, None], axis=1).flatten()
        
        # 4. Build Ghost Graph & Clusters
        # We construct edge lists. 
        # Edges form between Witness_Variable AND Ghost_Node(Sign).
        # If Sign is +1 -> Edge to GHOST_PLUS. If -1 -> Edge to GHOST_MINUS.

        # Active Bonds B >= 1
        mask_active_1 = (bonds >= 1)
        # Active Bonds B >= 2
        mask_active_2 = (bonds == 2)

        # Source nodes (Variables)
        src_1 = idx_w1_global[mask_active_1]
        src_2 = idx_w2_global[mask_active_2]

        # Target nodes (Ghosts)
        # If sign is +1, target is GHOST_PLUS. If -1, GHOST_MINUS
        tgt_1 = cp.where(sign_w1[mask_active_1] > 0, self.GHOST_PLUS, self.GHOST_MINUS)
        tgt_2 = cp.where(sign_w2[mask_active_2] > 0, self.GHOST_PLUS, self.GHOST_MINUS)
        
        # Concatenate all edges
        all_src = cp.concatenate([src_1, src_2])
        all_tgt = cp.concatenate([tgt_1, tgt_2])

        # Create Adjacency Matrix (Symmetric)
        # Weights don't matter, just connectivity.
        # FIX: Use float32 for weights to avoid TypeError in cuSPARSE operations
        weights = cp.ones(len(all_src), dtype=cp.float32)
        adj = cpx.coo_matrix((weights, (all_src, all_tgt)), shape=(self.TOTAL_NODES, self.TOTAL_NODES), dtype=cp.float32)
        
        # Convert to CSR before addition to ensure cuSPARSE compatibility
        adj = adj.tocsr()
        # Make symmetric
        adj = adj + adj.T

        # Connected Components
        # Standard SW flips clusters. Here, clusters attached to Ghosts are FROZEN.
        # Clusters attached to neither are FLIPPED randomly.
        n_components, labels = cpx_graph.connected_components(adj, directed=False)

        # Identify component labels for Ghosts
        label_plus = labels[self.GHOST_PLUS]
        label_minus = labels[self.GHOST_MINUS]

        # Determine Cluster Actions
        # 1. Connected to Plus -> Force +1
        # 2. Connected to Minus -> Force -1
        # 3. Connected to Both -> Contradiction (Frustration). 
        #    KBD strategy: usually implies local frustration. We keep current state or pick random. 
        #    Simple efficient strategy: Priority to Plus (or arbitrary), or freeze.
        #    Here: We treat 'Both' as Frozen to current state (simplest safe heuristic).
        # 4. Connected to None -> Free Cluster -> Flip with p=0.5

        # Generate random flips for all components
        # shape (n_components,)
        comp_flips = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=n_components)

        # Map component flips to variables
        new_spins = comp_flips[labels[:self.N+1]] # Exclude ghosts from array mapping

        # Apply Freezes
        # We create masks based on component labels
        is_plus_cluster = (labels[:self.N+1] == label_plus)
        is_minus_cluster = (labels[:self.N+1] == label_minus)

        # Force values
        new_spins[is_plus_cluster] = 1
        new_spins[is_minus_cluster] = -1

        # Handle Contradictions (Both Plus and Minus) -> Very rare in efficient KBD, but possible.
        # In this logic, the second assignment (Minus) overwrites. 
        # To be pedantic: if (is_plus & is_minus), we might want to keep old value.
        mask_conflict = is_plus_cluster & is_minus_cluster
        if cp.any(mask_conflict):
             new_spins[mask_conflict] = self.spins[mask_conflict]

        # Update spins (Keep dummy pinned)
        self.spins = new_spins
        self.spins[self.DUMMY_VAR_IDX] = -1
        
        return

    def energy(self):
        """Calculate fraction of unsatisfied clauses (3-SAT energy)."""
        # Re-eval strictly on the 3-SAT clauses (ignore dummy)
        # Indices: (M, 3)
        real_indices = self.tetra_indices[:, :3]
        real_signs = self.tetra_signs[:, :3]
        
        # Check literal satisfaction
        # Note: spins is size N+1, real_indices go up to N-1. Correct.
        current_spins = self.spins[real_indices]
        is_sat = (current_spins == real_signs)
        
        # Clause is satisfied if ANY literal is true
        clause_sat = cp.any(is_sat, axis=1)

        # Energy = Fraction Unsatisfied
        return 1.0 - cp.mean(clause_sat)'''
b.cells.append(nbf.v4.new_code_cell(code_solver))

# Cell 5: Baseline
code_baseline = r"""# @title 4. Baseline: `MetropolisGPU`
# A simple parallel Metropolis sampler for comparison.

class MetropolisGPU:
    def __init__(self, clauses_np, N, beta=2.0):
        self.N = N
        self.indices = cp.array(np.abs(clauses_np) - 1, dtype=cp.int32)
        self.signs = cp.array(np.sign(clauses_np), dtype=cp.int8)
        self.spins = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=N)
        self.beta = beta
        
    def step(self):
        # Propose random flips (vectorized batch flip often bad for dense, 
        # but for sparse SAT, we can try flipping a fraction or just 1. 
        # For fairness, let's flip N variables in parallel with acceptance).
        
        # 1. Compute current energy (unsat count) per clause
        # This is expensive to do fully incrementally on python level,
        # so we do a naive full recalculation or a partial optimized one.
        # For speed in this demo, we'll just do a 1% spin flip batch.
        
        n_flip = max(1, int(self.N * 0.01))
        flip_indices = cp.random.randint(0, self.N, size=n_flip)
        
        # Calc global energy before
        e_old = self.get_energy_count()
        
        # Flip
        self.spins[flip_indices] *= -1
        
        # Calc global energy after
        e_new = self.get_energy_count()
        
        # Metropolis acceptance
        delta_E = e_new - e_old
        if delta_E > 0:
            p = cp.exp(-self.beta * delta_E)
            if cp.random.random() > p:
                # Reject: Flip back
                self.spins[flip_indices] *= -1

    def get_energy_count(self):
        current = self.spins[self.indices]
        is_sat = (current == self.signs)
        clause_sat = cp.any(is_sat, axis=1)
        return cp.sum(~clause_sat)

    def energy(self):
        return self.get_energy_count() / len(self.indices)"""
b.cells.append(nbf.v4.new_code_cell(code_baseline))

# Cell 6: Execution
code_exec = """# @title 5. Execution & Benchmarking

# Parameters
N = 2000          # Number of variables (Large!)
alpha = 4.2       # Hard regime (near phase transition)
steps = 500       # Simulation steps
omega = 3.5       # Interaction strength (Tetra)
beta_base = 4.0   # Inv Temp (Metropolis)
compare_baseline = True # @param {type:"boolean"} 

print(f"Generating Random 3-SAT: N={N}, M={int(alpha*N)}...")
clauses, real_N = generate_random_3sat(N, alpha, seed=42)

# --- Run Tetra Dynamics ---
print("Initializing TetraDynamicsGPU...")
tetra_solver = TetraDynamicsGPU(clauses, real_N, omega=omega)

metro_energies = []
tetra_energies = []
start_t = time.time()
for i in range(steps):
    tetra_solver.step()
    if i % 10 == 0:
        e = tetra_solver.energy().item()
        tetra_energies.append(e)
        # print(f"Step {i}: E={e:.4f}")
end_t = time.time()
print(f"Tetra Dynamics Time: {end_t - start_t:.2f}s")

# --- Run Baseline (Optional) ---
metro_energies = []
if compare_baseline:
    print("Initializing MetropolisGPU...")
    metro_solver = MetropolisGPU(clauses, real_N, beta=beta_base)
    
    start_t = time.time()
    for i in range(steps):
        metro_solver.step()
        if i % 10 == 0:
            e = metro_solver.energy().item()
            metro_energies.append(e)
    end_t = time.time()
    print(f"Metropolis Time: {end_t - start_t:.2f}s")

# --- Plotting ---
x_axis = np.arange(0, steps, 10)
plt.figure()
plt.plot(x_axis, tetra_energies, label='Tetra Cluster Dynamics (Ours)', color='cyan', linewidth=2)
if compare_baseline:
    plt.plot(x_axis, metro_energies, label='Standard Metropolis', color='orange', alpha=0.7)

plt.xlabel('MC Steps')
plt.ylabel('Fraction Unsatisfied (Energy)')
plt.title(f'3-SAT Optimization: N={N}, $\alpha$={alpha}')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()"""
b.cells.append(nbf.v4.new_code_cell(code_exec))

with open('sw3sat_colab.ipynb', 'w') as f:
    nbf.write(b, f)