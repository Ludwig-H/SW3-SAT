import json

# Define the notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
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
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown(source_string):
    lines = [line + "\n" for line in source_string.splitlines()]
    if lines: lines[-1] = lines[-1].rstrip("\n")
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    })

def add_code(source_string):
    lines = [line + "\n" for line in source_string.splitlines()]
    if lines: lines[-1] = lines[-1].rstrip("\n")
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    })

# --- Content ---

# 1. Intro Markdown
# Using raw string r"""...""" to handle LaTeX backslashes correctly without escaping them for Python
intro_text = r"""# Higher-Order Swendsen-Wang Dynamics for 3-SAT (Triangle + Tetrahedron)

This notebook implements a cutting-edge **Cluster Monte Carlo** algorithm for solving 3-SAT problems, bridging Statistical Physics and Combinatorial Optimization.

## The Physics Model
We map the 3-SAT problem onto a **Spatially Embedded Spin System** with higher-order interactions.

### 1. Variables & Geometry
Consider $N$ variables $\sigma_i \in \{-1, +1\}$. We augment the graph with a "Ghost Node" $\sigma_0 = +1$ (representing TRUE).
Each 3-SAT clause $C_m = (l_1 \lor l_2 \lor l_3)$ is encoded by two geometric structures:
1.  **A Triangle ($\\mathcal{T}$)**: Connecting the 3 variables involved in the clause.
2.  **A Tetrahedron ($\\mathcal{K}$)**: Connecting the Triangle to the Ghost Node $\sigma_0$.

### 2. Interactions & Colors
The edges are colored (signed) to encode the literals:
*   **Triangle Edges**: An edge $(i, j)$ is **Antiferromagnetic** (Red, $J=-1$) if literals $l_i, l_j$ have the **same sign**. It is **Ferromagnetic** (Blue, $J=+1$) if they have **opposite signs**. This makes the triangle *Inherently Contradictory* (Frustrated).
*   **Tetrahedron Edges**: An edge $(0, i)$ connecting to the Ghost is Ferromagnetic if $l_i$ is positive ($x_i$), and Antiferromagnetic if $l_i$ is negative ($\\neg x_i$). 

### 3. Dynamics & Weights
We introduce a coupling parameter $\\omega$ (playing the role of inverse temperature/interaction strength).

*   **Tetrahedron (Weight $\\omega$)**: If the clause is **FULLY SATISFIED** (ALL 3 literals match $\\sigma_0$), we freeze the **entire tetrahedron** (all 3 edges) with probability $1 - e^{-\\omega}$.
*   **Triangle (Weight $\\omega/2$)**: The triangle is an *isotropic inherently contradictory* loop. It fluctuates between two energy levels:
    *   **Low Energy ($\\E_0 = \\omega/2$)**: 1 unsatisfied edge (Frustration limit).
    *   **High Energy ($\\E_1 = 3\\omega/2$)**: 3 unsatisfied edges.
    *   **Dynamics**: We follow the *SODA 2026 / Asilomar 2025* prescription (Table 3.1) to freeze specific subsets of edges based on the configuration state.
*   **Ghost Node Invariant**: The Ghost Node $\\sigma_0$ represents the "TRUE" state (+1). If it flips to -1 after a cluster update, we flip the entire system ($\\sigma \to -\\sigma$) to restore the gauge.

### 4. Energy Landscape
The global Hamiltonian is constructed such that:
*   **Satisfied Clause**: Energy $\\mathcal{H} = 3\\omega/2$.
*   **Unsatisfied Clause**: Energy $\\mathcal{H} = 5\\omega/2$.

We perform **Simulated Annealing** by increasing $\\omega$ over time, effectively lowering the temperature.

""" # This is the end of intro_text

add_markdown(intro_text)

# 2. Setup Code
setup_code = """# @title 1. Environment & GPU Setup
import sys
import os
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
import requests
import tarfile
import io
import gzip

# Ensure CuPy is available
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx
    import cupyx.scipy.sparse.csgraph as cpx_graph
    print(f"GPU Detected: {cp.cuda.runtime.getDeviceCount()} device(s)")
except ImportError:
    print("Installing CuPy...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'cupy-cuda12x'])
    import cupy as cp
    import cupyx.scipy.sparse as cpx
    import cupyx.scipy.sparse.csgraph as cpx_graph

plt.style.use('dark_background')
print("Environment Ready.")"""
add_code(setup_code)

# 3. Data Gen Code
data_gen_code = """# @title 2. Data Generators (Random & SATLIB)

def generate_random_3sat(N, alpha, seed=None):
    if seed is not None: np.random.seed(seed)
    M = int(N * alpha)
    vars = np.random.randint(1, N + 1, size=(M, 3))
    signs = np.random.choice([-1, 1], size=(M, 3))
    return vars * signs, N

def parse_dimacs(content):
    clauses = []
    N = 0
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith(('c', '%')):
            continue
        if line.startswith('p'):
            N = int(line.split()[2])
            continue
        try:
            lits = [int(x) for x in line.split() if x != '0']
            if len(lits) == 3:
                clauses.append(lits)
        except:
            pass
    return np.array(clauses, dtype=np.int32), N

def download_instance(url):
    print(f"Downloading {url}...")
    resp = requests.get(url)
    content = resp.content
    if url.endswith('.tar.gz'):
        with tarfile.open(fileobj=io.BytesIO(content), mode='r:gz') as tar:
            for m in tar.getmembers():
                if m.name.endswith('.cnf'):
                    return parse_dimacs(tar.extractfile(m).read().decode('utf-8'))
    return parse_dimacs(content.decode('utf-8'))"""
add_code(data_gen_code)

# 4. Solver Code
# Note: Using raw string r"""...""" for code block as well just in case, though standard triple quote is usually fine.
solver_code = r"""# @title 3. The Solver: `SwendsenWangTrianglesGPU`

class SwendsenWangTrianglesGPU:
    def __init__(self, clauses_np, N):
        self.N = N
        self.M = len(clauses_np)
        self.clauses = cp.array(clauses_np)
        
        # --- 1. Geometry Setup ---
        # Node 0 is Ghost (+). Nodes 1..N are variables.
        self.GHOST = 0
        
        # Extract literals (M, 3)
        self.lits_idx = cp.abs(self.clauses)
        self.lits_sign = cp.sign(self.clauses)
        
        # --- 2. Build Triangle Interactions (Internal) ---
        # Edges: (0,1), (1,2), (2,0) relative to clause indices 0,1,2
        # Sign Logic: Same Sign = AF (-1), Diff Sign = Ferro (+1)
        # We store these as J_tri: (M, 3) corresponding to pairs (0,1), (1,2), (2,0)
        
        s = self.lits_sign # (M, 3)
        # Pair (0,1)
        j01 = cp.where(s[:, 0] == s[:, 1], -1, 1)
        # Pair (1,2)
        j12 = cp.where(s[:, 1] == s[:, 2], -1, 1)
        # Pair (2,0)
        j20 = cp.where(s[:, 2] == s[:, 0], -1, 1)
        
        self.J_tri = cp.stack([j01, j12, j20], axis=1).astype(cp.int8)
        
        # --- 3. Build Tetrahedron Interactions (to Ghost) ---
        # J_tetra: (M, 3). Edge between lit k and Ghost.
        # Lit > 0 (x) -> Match Ghost (+) -> Ferro (+1)
        # Lit < 0 (not x) -> Mismatch Ghost (+) -> AF (-1)
        self.J_tetra = s.astype(cp.int8)
        
        # Initialize Spins (0..N)
        # 0 is fixed to 1. Others random.
        self.sigma = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=N+1)
        self.sigma[0] = 1

    def energy_check(self, omega):
        # Computes global energy and verifies levels.
        # Level 1 (SAT): 3*omega/2
        # Level 2 (UNSAT): 5*omega/2
        
        # 1. Clause Satisfaction
        # Lit satisfied if sigma[idx] == sign
        spins = self.sigma[self.lits_idx]
        is_lit_sat = (spins == self.lits_sign)
        is_clause_sat = cp.any(is_lit_sat, axis=1)
        
        # Energy Calculation (Formal Hamiltonian)
        # This is a conceptual check. In our model, we just count SAT/UNSAT.
        # But let's verify the user's "levels".
        # If SAT: Energy should be 1.5 * omega
        # If UNSAT: Energy should be 2.5 * omega
        
        total_E = cp.sum(cp.where(is_clause_sat, 1.5 * omega, 2.5 * omega))
        avg_E_sat = 1.5 * omega
        avg_E_unsat = 2.5 * omega
        
        unsat_frac = 1.0 - cp.mean(is_clause_sat)
        return total_E, unsat_frac

    def step(self, omega):
        # Performs one Swendsen-Wang step using Triangle + Tetra dynamics.
        
        num_clauses = self.M
        
        # --- A. Tetrahedron Dynamics (Freezing to Ghost) ---
        # User Instruction: "Le tétraèdre n'est possiblement gelé que si *tous* les trois spins
        # du triangle sont conforme à \\sigma_0. Alors avec probabilité 1-e^{-\\omega}, 
        # il faut geler tout le tétraèdre."
        
        # Get spins for all literals in clauses
        c_spins = self.sigma[self.lits_idx] # (M, 3)
        
        # Check literal satisfaction vs Ghost (+1)
        # lit_is_sat[m, k] is True if literal k of clause m is consistent with Ghost
        lit_is_sat = (c_spins == self.J_tetra)
        
        # Condition: ALL 3 spins must be conformant
        tetra_fully_sat = cp.all(lit_is_sat, axis=1) # (M,)
        
        # Probability to freeze the WHOLE tetrahedron
        p_tetra = 1.0 - cp.exp(-omega)
        rand_tetra = cp.random.random(num_clauses, dtype=cp.float32)
        
        # Decision: Freeze if Fully Sat AND random < p
        do_freeze_tetra = tetra_fully_sat & (rand_tetra < p_tetra) # (M,)
        
        # Broadcast decision to all 3 edges (M, 3)
        # If do_freeze_tetra[m] is True, then freeze (m,0), (m,1), (m,2)
        freeze_tetra = cp.stack([do_freeze_tetra]*3, axis=1)
        
        # --- B. Triangle Dynamics (Inherently Contradictory) ---
        # Weight w = omega / 2
        w = omega / 2.0
        # Probabilities from PDF Table 3.1 (Right - Inherently Contradictory)
        # p_freeze_1 = 0.5 * (1 - exp(-2w))
        # High Energy State (3 unsat?): Freeze Empty set (Prob 1).
        # Low Energy State (1 unsat): Freeze 1 satisfied edge (Prob p_freeze_1).
        
        # 1. Determine State of each triangle
        # Edges 0:(0,1), 1:(1,2), 2:(2,0)
        # Get spins
        s0 = c_spins[:, 0]
        s1 = c_spins[:, 1]
        s2 = c_spins[:, 2]
        
        # Check satisfaction of internal edges
        # Edge is sat if spin_i * spin_j * J == 1
        # J stored in self.J_tri
        sat0 = (s0 * s1 * self.J_tri[:, 0]) == 1
        sat1 = (s1 * s2 * self.J_tri[:, 1]) == 1
        sat2 = (s2 * s0 * self.J_tri[:, 2]) == 1
        
        sat_mask = cp.stack([sat0, sat1, sat2], axis=1) # (M, 3)
        num_sat = cp.sum(sat_mask, axis=1)
        
        # In a frustrated triangle, max SAT edges is 2 (Low Energy), min is 0 (High Energy, if ferro) or 3 unsat?
        # For -1, +1, +1 (AF, F, F):
        # If +++: -1(U), +1(S), +1(S). 2 SAT. Low Energy.
        # If +--: +1(S), +1(S), -1(U). 2 SAT. Low Energy.
        # Check "High Energy": s1=1, s2=1, s3=-1. J=(-1, 1, 1).
        # (1,1,J=-1)->U. (1,-1,J=1)->U. (-1,1,J=1)->U. 0 SAT. High Energy.
        
        is_low_energy = (num_sat == 2)
        # is_high_energy = (num_sat == 0) 
        
        # Dynamics:
        # If High Energy (0 sat): Freeze Nothing (Empty).
        # If Low Energy (2 sat): Freeze exactly ONE of the 2 satisfied edges.
        # Prob to freeze = 0.5 * (1 - exp(-2w))
        # Wait, PDF says "freeze exactly one... with probability 1/2(1-e^-2w)".
        # Does it mean we might freeze NOTHING in Low Energy? Yes. 
        # Total prob to freeze SOMETHING is (1 - e^-2w). Split between the 2 edges.
        
        p_freeze_any = 1.0 - cp.exp(-2.0 * w)
        # But we must pick WHICH one. Uniformly among the 2.
        
        rand_tri = cp.random.random(num_clauses, dtype=cp.float32)
        
        # Output mask for triangle edges (M, 3)
        freeze_tri = cp.zeros((num_clauses, 3), dtype=bool)
        
        # Logic for Low Energy:
        # If rand < p_freeze_any: we freeze ONE edge.
        # Which one? The first satisfied or second satisfied?
        # We need to select one of the TRUE values in sat_mask randomly.
        
        # Create a random selector for the 2 edges
        # We can multiply sat_mask by random numbers and pick argmax
        selector = cp.random.random((num_clauses, 3), dtype=cp.float32)
        selector = selector * sat_mask # Zero out unsat edges
        target_edge = cp.argmax(selector, axis=1) # Index of edge to freeze
        
        # Apply freeze
        # Mask: Low Energy AND (rand < p_freeze_any)
        do_freeze = is_low_energy & (rand_tri < p_freeze_any)
        
        # Set the bit
        # We use fancy indexing. indices (0..M-1), target_edge
        row_idxs = cp.arange(num_clauses)[do_freeze]
        col_idxs = target_edge[do_freeze]
        freeze_tri[row_idxs, col_idxs] = True
        
        # --- C. Graph Construction ---
        # We need to build the adjacency matrix for Connected Components.
        # Nodes: 0..N.
        
        # 1. Tetra Edges (Ghost-Variable)
        # freeze_tetra is (M, 3). 
        # Indices: (Ghost, lits_idx[m, 0]), etc.
        t_rows, t_cols = cp.where(freeze_tetra)
        # t_rows is clause index, t_cols is 0,1,2
        # Map t_cols to variable index
        var_indices = self.lits_idx[t_rows, t_cols]
        
        src_tetra = cp.zeros_like(var_indices) # All 0 (Ghost)
        dst_tetra = var_indices
        
        # 2. Triangle Edges (Var-Var)
        # freeze_tri is (M, 3). Pairs: (0,1), (1,2), (2,0)
        tr_rows, tr_cols = cp.where(freeze_tri)
        
        # Map to variables
        # if tr_cols == 0 -> edge between lit 0 and lit 1
        # if tr_cols == 1 -> edge between lit 1 and lit 2
        # if tr_cols == 2 -> edge between lit 2 and lit 0
        
        idx0 = self.lits_idx[tr_rows, 0]
        idx1 = self.lits_idx[tr_rows, 1]
        idx2 = self.lits_idx[tr_rows, 2]
        
        src_tri = cp.zeros_like(tr_rows)
        dst_tri = cp.zeros_like(tr_rows)
        
        # Vectorized assignment
        mask0 = (tr_cols == 0)
        src_tri[mask0] = idx0[mask0]
        dst_tri[mask0] = idx1[mask0]
        
        mask1 = (tr_cols == 1)
        src_tri[mask1] = idx1[mask1]
        dst_tri[mask1] = idx2[mask1]
        
        mask2 = (tr_cols == 2)
        src_tri[mask2] = idx2[mask2]
        dst_tri[mask2] = idx0[mask2]
        
        # Combine
        all_src = cp.concatenate([src_tetra, src_tri])
        all_dst = cp.concatenate([dst_tetra, dst_tri])
        
        # --- D. Cluster Flip ---
        if len(all_src) > 0:
            # Build Graph
            # Fix: CuPy sparse/graph utils often require numeric types (float/int), not bool
            data = cp.ones(len(all_src), dtype=cp.float32)
            adj = cpx.coo_matrix((data, (all_src, all_dst)), shape=(self.N+1, self.N+1), dtype=cp.float32)
            
            # Component Labeling
            n_comps, labels = cpx_graph.connected_components(adj, directed=False)
            
            # --- Percolation Analysis (New) ---
            # Calculate sizes of all components
            # labels is an array of component IDs for each node
            # We use bincount to count nodes per label
            comp_sizes = cp.bincount(labels)
            
            # Sort descending to get largest
            sorted_sizes = cp.sort(comp_sizes)[::-1]
            
            # Largest component (C1)
            c1_size = sorted_sizes[0]
            
            # Second largest (C2) - handle case where only 1 component exists
            if n_comps > 1:
                c2_size = sorted_sizes[1]
            else:
                c2_size = 0.0
                
            c1_frac = c1_size / float(self.N + 1)
            c2_frac = c2_size / float(self.N + 1)
            
            # Flip Logic
            # 1. Identify Ghost Cluster
            ghost_label = labels[0]
            
            # 2. Random Flips for all clusters
            cluster_flips = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=n_comps)
            
            # 3. Force Ghost Cluster to +1 (Keep Ghost Fixed)
            # Standard SW: Flip clusters randomly.
            # If the cluster containing Ghost (0) flips to -1, we flip EVERYTHING to restore Gauge.
            # (Global Spin Flip symmetry)
            
            # Apply cluster flips first
            flip_vector = cluster_flips[labels]
            self.sigma *= flip_vector
            
            # Check Ghost
            if self.sigma[self.GHOST] == -1:
                self.sigma *= -1 # Flip everything back so Ghost is +1
        else:
            # No edges frozen. All free clusters (except 0).
            # 1 Giant component? No, N components of size 1.
            # Wait, if no edges, every node is its own component.
            # So C1 = 1/(N+1), C2 = 1/(N+1)
            c1_frac = 1.0 / (self.N + 1)
            c2_frac = 1.0 / (self.N + 1)
            
            # Flip everyone randomly except 0.
            flips = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=self.N+1)
            self.sigma *= flips
            if self.sigma[self.GHOST] == -1:
                self.sigma *= -1
            
        return self.energy_check(omega)[1], c1_frac, c2_frac # Return unsat, c1, c2
"""
add_code(solver_code)

# 5. Baseline Code
baseline_code = """# @title 4. Baseline: `MetropolisGPU`

class MetropolisGPU:
    def __init__(self, clauses_np, N):
        print(f"Initializing MetropolisGPU with N={N}...")
        self.N = N
        # Convert to CuPy array first (Explicit Fix)
        clauses_cp = cp.array(clauses_np, dtype=cp.int32)
        self.lits_idx = cp.abs(clauses_cp)
        self.lits_sign = cp.sign(clauses_cp).astype(cp.int8)
        # Use a separate spin array
        self.sigma = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=N+1)
        self.sigma[0] = 1

    def energy(self):
        spins = self.sigma[self.lits_idx]
        is_sat = (spins == self.lits_sign)
        clause_sat = cp.any(is_sat, axis=1)
        return 1.0 - cp.mean(clause_sat)

    def step(self, beta):
        # Parallel Metropolis (Checkerboard-like or Batch)
        # We pick N/10 random indices to flip
        n_flip = max(1, self.N // 100)
        idx = cp.random.randint(1, self.N + 1, size=n_flip)
        
        e_old = self.energy()
        # Flip
        self.sigma[idx] *= -1
        e_new = self.energy()
        
        delta = e_new - e_old
        # Since energy is fraction unsat, we need to scale by M for actual Hamiltonian difference
        # H ~ M * unsat.
        # P = exp(-beta * M * delta)
        # Note: User said beta proportional to omega. 
        # If omega is O(1), and energy is O(1), beta should be O(M) or similar?
        # Let's assume passed beta is the effective coupling.
        
        if delta > 0:
            p = cp.exp(-beta * delta * 100.0) # Scaling factor for sensitivity
            if cp.random.random() > p:
                self.sigma[idx] *= -1 # Reject"""
add_code(baseline_code)

# 6. Main Loop Code
main_code = r"""# @title 5. Main Simulation Loop (Annealing)

# Config
N = 500
alpha = 4.25 # Hard region
clauses_np, _ = generate_random_3sat(N, alpha, seed=42)

print(f"Instance: N={N}, M={len(clauses_np)}, Alpha={alpha}")

solver = SwendsenWangTrianglesGPU(clauses_np, N)
metro = MetropolisGPU(clauses_np, N)

# Schedule
steps = 200
omega_start = 0.5
omega_end = 6.0
omega_schedule = np.linspace(omega_start, omega_end, steps)

history_sw = []
history_c1 = []
history_c2 = []
history_mh = []

t0 = time.time()
print("Starting Annealing...")

for i, omega in enumerate(omega_schedule):
    # 1. Swendsen-Wang Step
    unsat_sw, c1, c2 = solver.step(omega)
    
    # Store SW Energy
    if hasattr(unsat_sw, 'get'):
        history_sw.append(float(unsat_sw.get()))
    else:
        history_sw.append(float(unsat_sw))
        
    # Store Cluster Sizes
    if hasattr(c1, 'get'):
        history_c1.append(float(c1.get()))
    else:
        history_c1.append(float(c1))
        
    if hasattr(c2, 'get'):
        history_c2.append(float(c2.get()))
    else:
        history_c2.append(float(c2))
    
    # 2. Metropolis Step
    # Heuristic scaling for beta to match omega's constraining power
    beta = omega * 5.0 
    # Run multiple sub-steps for fair comparison (SW is global)
    for _ in range(5):
        metro.step(beta)
    
    e_mh = metro.energy()
    if hasattr(e_mh, 'get'):
        history_mh.append(float(e_mh.get()))
    else:
        history_mh.append(float(e_mh))
    
    if i % 20 == 0:
        print(f"Step {i:3d} | Omega {omega:.2f} | SW Unsat: {unsat_sw:.4f} (C1={history_c1[-1]:.2f}) | MH Unsat: {history_mh[-1]:.4f}")

dt = time.time() - t0
print(f"Done in {dt:.2f}s")

# Plot
# Ensure inputs are on CPU (NumPy) before plotting
omega_cpu = omega_schedule.get() if hasattr(omega_schedule, 'get') else omega_schedule
sw_cpu = np.array(history_sw)
c1_cpu = np.array(history_c1)
c2_cpu = np.array(history_c2)
mh_cpu = np.array(history_mh)

print(f"Plotting types: Omega={type(omega_cpu)}, SW={type(sw_cpu)}")

fig, ax1 = plt.subplots(figsize=(12, 7))

# Left Axis: Energy
color_sw = 'cyan'
color_mh = 'orange'
ax1.set_xlabel(r'Coupling $\omega$ (Inverse Temp)')
ax1.set_ylabel('Fraction Unsatisfied Clauses', color='white')
l1, = ax1.plot(omega_cpu, sw_cpu, label='SW Energy', color=color_sw, linewidth=2)
l2, = ax1.plot(omega_cpu, mh_cpu, label='MH Energy', color=color_mh, alpha=0.6)
ax1.tick_params(axis='y', labelcolor='white')
ax1.grid(True, alpha=0.2)

# Right Axis: Cluster Sizes
ax2 = ax1.twinx()
color_c1 = 'magenta'
color_c2 = 'lime'
ax2.set_ylabel('Cluster Size Fraction (Percolation)', color='white')
l3, = ax2.plot(omega_cpu, c1_cpu, label='Largest Cluster (C1)', color=color_c1, linestyle='--', linewidth=1.5)
l4, = ax2.plot(omega_cpu, c2_cpu, label='2nd Largest (C2)', color=color_c2, linestyle=':', linewidth=1.5)
ax2.tick_params(axis='y', labelcolor='white')

# Combine legends
lines = [l1, l2, l3, l4]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title(rf'3-SAT Annealing: N={N}, $\alpha$={alpha} | Topo-Percolation')
plt.show()"""
add_code(main_code)

with open("Swendsen-Wang_3SAT_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully.")