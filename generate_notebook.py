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
intro_text = r"""# Stochastic Higher-Order Swendsen-Wang Dynamics for 3-SAT

This notebook implements an advanced **Stochastic Cluster Monte Carlo** algorithm.
It combines global cluster moves (Swendsen-Wang) with local heuristics derived from UNSAT clauses (Focusing).

## The Algorithm
1.  **Marking**: Variables involved in UNSAT clauses are "marked".
2.  **Hybrid Dynamics**:
    *   **Tetrahedrons (Fully SAT)**: Connect Ghost to UNMARKED variables. If all marked, connect to one random variable.
    *   **Triangles (Low Energy)**:
        *   Behavior depends on how many vertices are marked (0, 1, 2, 3).
        *   Generally avoids freezing edges between marked variables.
        *   Tries to connect satisfied literals to Ghost to stabilize them.
3.  **Percolation & Flip**: Standard cluster flip step.

"""
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
"""
add_code(data_gen_code)

# 4. Stochastic Solver Code
solver_code = r"""# @title 3. The Solver: `StochasticSwendsenWangGPU`

class StochasticSwendsenWangGPU:
    def __init__(self, clauses_np, N):
        self.N = N
        self.M = len(clauses_np)
        self.clauses = cp.array(clauses_np)
        self.GHOST = 0
        
        # Literals
        self.lits_idx = cp.abs(self.clauses)
        self.lits_sign = cp.sign(self.clauses)
        
        # Interactions
        s = self.lits_sign
        j01 = cp.where(s[:, 0] == s[:, 1], -1, 1)
        j12 = cp.where(s[:, 1] == s[:, 2], -1, 1)
        j20 = cp.where(s[:, 2] == s[:, 0], -1, 1)
        self.J_tri = cp.stack([j01, j12, j20], axis=1).astype(cp.int8)
        self.J_tetra = s.astype(cp.int8)
        
        # State
        self.sigma = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=N+1)
        self.sigma[0] = 1

    def energy_check(self, omega):
        spins = self.sigma[self.lits_idx]
        is_lit_sat = (spins == self.lits_sign)
        is_clause_sat = cp.any(is_lit_sat, axis=1)
        unsat_frac = 1.0 - cp.mean(is_clause_sat)
        return unsat_frac

    def step(self, omega):
        # 1. Calculate Clause Status
        c_spins = self.sigma[self.lits_idx]
        lit_is_sat = (c_spins == self.J_tetra)
        num_lit_sat = cp.sum(lit_is_sat, axis=1)
        
        is_fully_sat = (num_lit_sat == 3)
        is_unsat = (num_lit_sat == 0) # High Energy / UNSAT Clause
        
        # Triangle Internal Status
        s0, s1, s2 = c_spins[:, 0], c_spins[:, 1], c_spins[:, 2]
        sat0 = (s0 * s1 * self.J_tri[:, 0] == 1)
        sat1 = (s1 * s2 * self.J_tri[:, 1] == 1)
        sat2 = (s2 * s0 * self.J_tri[:, 2] == 1)
        sat_mask = cp.stack([sat0, sat1, sat2], axis=1)
        num_sat_tri = cp.sum(sat_mask, axis=1)
        
        # Low Energy Triangle = 2 satisfied edges (occurs when 1 or 2 lits sat)
        # Note: In our signed construction, Fully SAT (3 lits) also implies Low Energy (2 edges).
        # But we handle Fully SAT separately in Tetra logic.
        is_low_energy = (num_sat_tri == 2)

        # 2. Marking Step
        # Mark variables involved in UNSAT clauses
        # is_unsat is boolean (M,)
        # We need a boolean mask for variables (N+1)
        
        marked_vars = cp.zeros(self.N + 1, dtype=bool)
        if cp.any(is_unsat):
            unsat_vars = self.lits_idx[is_unsat].flatten()
            marked_vars[unsat_vars] = True
            
        # Get Marked Status per Clause Literal
        # (M, 3) boolean
        lit_marked = marked_vars[self.lits_idx]
        num_marked = cp.sum(lit_marked, axis=1) # 0, 1, 2, or 3
        
        # 3. Randomness
        P = 1.0 - cp.exp(-omega)
        # We need a uniform random variable per clause to decide actions
        rand_vals = cp.random.random(self.M, dtype=cp.float32)
        
        src_nodes = []
        dst_nodes = []
        
        # --- A. Tetrahedron Logic (Fully SAT) ---
        # Clause is Satisfied (3 lits).
        # Condition: is_fully_sat
        
        mask_A = is_fully_sat & (rand_vals < P)
        if cp.any(mask_A):
            idx_A = cp.where(mask_A)[0]
            
            # Sub-masks for marked count within A
            n_marked_A = num_marked[idx_A]
            
            # Case A1: 3 Marked (All marked) -> Link Ghost to ONE random vertex
            mask_A1 = (n_marked_A == 3)
            if cp.any(mask_A1):
                idx_A1 = idx_A[mask_A1]
                # Pick one random lit (0, 1, 2)
                # We can reuse rand_vals or gen new. Let's gen small new for selection
                # Or use modulo of current rand? Cleaner to gen new.
                r_sel = cp.random.randint(0, 3, size=len(idx_A1))
                targets = self.lits_idx[idx_A1, r_sel]
                src_nodes.append(cp.zeros_like(targets)) # Ghost
                dst_nodes.append(targets)
            
            # Case A2: Not all marked (< 3) -> Link Ghost to ALL UNMARKED vertices
            mask_A2 = (n_marked_A < 3)
            if cp.any(mask_A2):
                idx_A2 = idx_A[mask_A2]
                # Identify unmarked literals in these clauses
                # lit_marked[idx_A2] is (K, 3) bool
                # We want indices where lit_marked is False
                unmarked_mask = ~lit_marked[idx_A2] # (K, 3)
                
                # We need to extract these.
                # Use where on the mask
                rows, cols = cp.where(unmarked_mask)
                # Map back to global clause indices
                clause_indices = idx_A2[rows]
                # Get variable indices
                targets = self.lits_idx[clause_indices, cols]
                
                src_nodes.append(cp.zeros_like(targets))
                dst_nodes.append(targets)

        # --- B. Triangle Logic (Low Energy & NOT Fully Sat) ---
        # Condition: is_low_energy & (~is_fully_sat)
        mask_B = is_low_energy & (~is_fully_sat) & (rand_vals < P)
        
        if cp.any(mask_B):
            idx_B = cp.where(mask_B)[0]
            n_marked_B = num_marked[idx_B]
            
            # Sub-logic based on marked count
            
            # --- Case B3: 3 Marked ---
            # Action: Link Ghost to ONE random SATISFIED literal
            # We need to identify satisfied literals first.
            # In Low Energy (Not Fully Sat), we have 1 or 2 satisfied literals.
            mask_B3 = (n_marked_B == 3)
            if cp.any(mask_B3):
                idx_B3 = idx_B[mask_B3]
                # Get sat status (K, 3)
                sat_lits_B3 = lit_is_sat[idx_B3]
                
                # We need to pick one True value per row randomly
                # Trick: Multiply by random, pick argmax
                r_sel = cp.random.random(sat_lits_B3.shape, dtype=cp.float32)
                r_sel = r_sel * sat_lits_B3 # Zero out unsat
                chosen_col = cp.argmax(r_sel, axis=1)
                
                targets = self.lits_idx[idx_B3, chosen_col]
                src_nodes.append(cp.zeros_like(targets))
                dst_nodes.append(targets)

            # --- Case B2: 2 Marked ---
            mask_B2 = (n_marked_B == 2)
            if cp.any(mask_B2):
                idx_B2 = idx_B[mask_B2]
                # Identify the single Unmarked vertex (col index)
                # lit_marked[idx_B2] has two True and one False
                unmarked_col = cp.argmin(lit_marked[idx_B2], axis=1) # argmin of bool gives index of False
                
                # Check if Unmarked vertex is Satisfied
                # lit_is_sat[idx_B2, unmarked_col]
                # We need fancy indexing
                row_ids = cp.arange(len(idx_B2))
                is_unmarked_sat = lit_is_sat[idx_B2, unmarked_col]
                
                # Subcase B2.1: Unmarked is SAT -> Link Ghost to Unmarked
                if cp.any(is_unmarked_sat):
                    sub_idx = row_ids[is_unmarked_sat] # Local indices
                    real_idx = idx_B2[sub_idx]
                    cols = unmarked_col[sub_idx]
                    
                    targets = self.lits_idx[real_idx, cols]
                    src_nodes.append(cp.zeros_like(targets))
                    dst_nodes.append(targets)
                    
                # Subcase B2.2: Unmarked is UNSAT -> Freeze one internal SAT edge, NOT connecting the 2 marked
                # The 2 marked vertices are at indices != unmarked_col
                # The edge connecting the two marked vertices is opposite to unmarked_col.
                # Edge 0 connects (0,1), Edge 1 connects (1,2), Edge 2 connects (2,0).
                # If unmarked is 2, marked are 0,1. Edge 0 connects them.
                # We must FORBID Edge = unmarked_col.
                # We must choose a Satisfied Edge that is NOT unmarked_col.
                
                is_unmarked_unsat = ~is_unmarked_sat
                if cp.any(is_unmarked_unsat):
                    sub_idx = row_ids[is_unmarked_unsat]
                    real_idx = idx_B2[sub_idx]
                    forbidden_edge = unmarked_col[sub_idx] # This is the edge index connecting the two marked vars?
                    # Wait: Edge 0 connects l0-l1. If l2 is unmarked (so l0, l1 marked), Edge 0 connects marked.
                    # So yes, forbidden_edge index == unmarked_vertex index.
                    
                    # Available edges: {0,1,2} \ {forbidden}
                    # Among available, we need a Satisfied one.
                    # In Low Energy, 2 edges are Sat, 1 Unsat.
                    # We just need to find a Sat edge != forbidden.
                    
                    # Global sat mask for these clauses
                    c_sat_mask = sat_mask[real_idx] # (K, 3)
                    
                    # Mask out forbidden
                    # We can clone and set forbidden to False
                    temp_mask = c_sat_mask.copy()
                    temp_mask[cp.arange(len(real_idx)), forbidden_edge] = False
                    
                    # Now pick any True edge index
                    # Since it's Low Energy, there should be at least one remaining (unless the only 2 sat edges were... wait)
                    # If 2 edges are sat. 1 is forbidden. Is it possible the other sat is also forbidden? No, 1 forbidden.
                    # Is it possible the *only* sat edges are forbidden? No, logic says 2 sat edges. We forbid 1. At least 1 left.
                    
                    target_edge = cp.argmax(temp_mask, axis=1)
                    
                    # Convert edge to vars
                    lits = self.lits_idx[real_idx]
                    l0, l1, l2 = lits[:,0], lits[:,1], lits[:,2]
                    
                    s_e = cp.where(target_edge==0, l0, cp.where(target_edge==1, l1, l2))
                    d_e = cp.where(target_edge==0, l1, cp.where(target_edge==1, l2, l0))
                    src_nodes.append(s_e)
                    dst_nodes.append(d_e)

            # --- Case B1: 1 Marked ---
            mask_B1 = (n_marked_B == 1)
            if cp.any(mask_B1):
                idx_B1 = idx_B[mask_B1]
                # Identify Marked vertex (col index)
                marked_col = cp.argmax(lit_marked[idx_B1], axis=1)
                
                # Check Edge opposite to Marked (index == marked_col)
                # Is it Satisfied?
                # sat_mask[idx_B1, marked_col]
                row_ids = cp.arange(len(idx_B1))
                is_opp_sat = sat_mask[idx_B1, marked_col]
                
                # Subcase B1.1: Opp Edge SAT -> Freeze it
                if cp.any(is_opp_sat):
                    sub_idx = row_ids[is_opp_sat]
                    real_idx = idx_B1[sub_idx]
                    target_edge = marked_col[sub_idx]
                    
                    lits = self.lits_idx[real_idx]
                    l0, l1, l2 = lits[:,0], lits[:,1], lits[:,2]
                    s_e = cp.where(target_edge==0, l0, cp.where(target_edge==1, l1, l2))
                    d_e = cp.where(target_edge==0, l1, cp.where(target_edge==1, l2, l0))
                    src_nodes.append(s_e)
                    dst_nodes.append(d_e)
                
                # Subcase B1.2: Opp Edge UNSAT
                is_opp_unsat = ~is_opp_sat
                if cp.any(is_opp_unsat):
                    sub_idx = row_ids[is_opp_unsat]
                    real_idx = idx_B1[sub_idx]
                    m_col = marked_col[sub_idx]
                    
                    # Check Marked Vertex Sat Status
                    is_marked_lit_sat = lit_is_sat[real_idx, m_col]
                    
                    # B1.2.a: Marked Lit is UNSAT (=> The other 2 are SAT, since Low Energy usually implies 1 or 2 sat lits)
                    # If Marked is Unsat, and Opp Edge is Unsat... wait.
                    # Triangle Logic: J1*J2*J3 = -1.
                    # Spins: M(Unsat), A(Sat), B(Sat).
                    # Check consistency.
                    # User says: "Link Ghost to one of the two others (randomly)"
                    
                    mask_a = (~is_marked_lit_sat)
                    if cp.any(mask_a):
                        # Pick one of the other 2 cols
                        # The other cols are (m_col+1)%3 and (m_col+2)%3
                        idx_a = real_idx[mask_a]
                        mc = m_col[mask_a]
                        
                        r_choice = cp.random.randint(0, 2, size=len(idx_a)) # 0 or 1
                        # If 0 -> +1, If 1 -> +2
                        offset = r_choice + 1
                        target_col = (mc + offset) % 3
                        
                        targets = self.lits_idx[idx_a, target_col]
                        src_nodes.append(cp.zeros_like(targets))
                        dst_nodes.append(targets)
                        
                    # B1.2.b: Marked Lit is SAT
                    # User says: "Link Ghost to Marked Lit"
                    mask_b = (is_marked_lit_sat)
                    if cp.any(mask_b):
                        idx_b = real_idx[mask_b]
                        mc = m_col[mask_b]
                        targets = self.lits_idx[idx_b, mc]
                        src_nodes.append(cp.zeros_like(targets))
                        dst_nodes.append(targets)

            # --- Case B0: 0 Marked ---
            # Standard Swendsen-Wang Triangle Step
            mask_B0 = (n_marked_B == 0)
            if cp.any(mask_B0):
                idx_B0 = idx_B[mask_B0]
                # Pick one of the 2 satisfied edges randomly
                sub_sat = sat_mask[idx_B0] # (K, 3)
                
                # Random selection logic
                # P/2 split is already handled by 'rand_vals < P'? No, P is global.
                # We need to split the population of B0 into "Pick 1st" and "Pick 2nd"
                # using the existing random numbers r_sub corresponding to these?
                # User logic: "Randomly pick one".
                
                # Reuse rand_vals
                r_vals = rand_vals[mask_B][mask_B0]
                # Re-normalize to 0..1 range? Or just parity?
                # Let's check bit 0 or just < 0.5 * P (approx)
                # Since rand < P, and P is small, we can check < P/2
                
                pick_first = (r_vals < (P / 2.0))
                
                idx_1st = cp.argmax(sub_sat, axis=1)
                
                temp = sub_sat.copy()
                row_ids = cp.arange(len(idx_B0))
                temp[row_ids, idx_1st] = False
                idx_2nd = cp.argmax(temp, axis=1)
                
                chosen_edge_idx = cp.where(pick_first, idx_1st, idx_2nd)
                
                lits = self.lits_idx[idx_B0]
                l0, l1, l2 = lits[:,0], lits[:,1], lits[:,2]
                s_e = cp.where(chosen_edge_idx==0, l0, cp.where(chosen_edge_idx==1, l1, l2))
                d_e = cp.where(chosen_edge_idx==0, l1, cp.where(chosen_edge_idx==1, l2, l0))
                src_nodes.append(s_e)
                dst_nodes.append(d_e)

        # --- 4. Cluster & Flip ---
        # Initialize percolation metrics to avoid UnboundLocalError
        c1_frac = 0.0
        c2_frac = 0.0

        if len(src_nodes) > 0:
            all_src = cp.concatenate(src_nodes)
            all_dst = cp.concatenate(dst_nodes)
            
            data = cp.ones(len(all_src), dtype=cp.float32)
            adj = cpx.coo_matrix((data, (all_src, all_dst)), shape=(self.N+1, self.N+1), dtype=cp.float32)
            n_comps, labels = cpx_graph.connected_components(adj, directed=False)
            
            # Flip Logic
            ghost_label = labels[0]
            cluster_flips = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=n_comps)
            
            flip_vector = cluster_flips[labels]
            self.sigma *= flip_vector
            
            if self.sigma[self.GHOST] == -1:
                self.sigma *= -1 
        else:
            c1_frac = 1.0 / (self.N + 1)
            c2_frac = 1.0 / (self.N + 1)
            
            flips = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=self.N+1)
            self.sigma *= flips
            if self.sigma[self.GHOST] == -1:
                self.sigma *= -1
            
        return self.energy_check(omega), c1_frac, c2_frac
"""
add_code(solver_code)

# 5. Baseline (Metropolis)
baseline_code = """# @title 4. Baseline: `MetropolisGPU`
class MetropolisGPU:
    def __init__(self, clauses_np, N):
        self.N = N
        clauses_cp = cp.array(clauses_np, dtype=cp.int32)
        self.lits_idx = cp.abs(clauses_cp)
        self.lits_sign = cp.sign(clauses_cp).astype(cp.int8)
        self.sigma = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=N+1)
        self.sigma[0] = 1

    def energy(self):
        spins = self.sigma[self.lits_idx]
        is_sat = (spins == self.lits_sign)
        clause_sat = cp.any(is_sat, axis=1)
        return 1.0 - cp.mean(clause_sat)

    def step(self, beta):
        n_flip = max(1, self.N // 100)
        idx = cp.random.randint(1, self.N + 1, size=n_flip)
        e_old = self.energy()
        self.sigma[idx] *= -1
        e_new = self.energy()
        delta = e_new - e_old
        if delta > 0:
            p = cp.exp(-beta * delta * 100.0)
            if cp.random.random() > p:
                self.sigma[idx] *= -1
"""
add_code(baseline_code)

# 6. Main Loop
main_code = r"""# @title 5. Main Simulation Loop
N = 500
alpha = 4.25
clauses_np, _ = generate_random_3sat(N, alpha, seed=42)
print(f"Instance: N={N}, M={len(clauses_np)}, Alpha={alpha}")

# Use the New Solver
solver = StochasticSwendsenWangGPU(clauses_np, N)
metro = MetropolisGPU(clauses_np, N)

steps = 200
omega_schedule = np.linspace(0.5, 6.0, steps)

history_sw = []
history_c1 = []
history_c2 = []
history_mh = []

t0 = time.time()
print("Starting Annealing...")

for i, omega in enumerate(omega_schedule):
    # Stochastic SW Step
    unsat_sw, c1, c2 = solver.step(omega)
    
    if hasattr(unsat_sw, 'get'): history_sw.append(float(unsat_sw.get()))
    else: history_sw.append(float(unsat_sw))
    
    if hasattr(c1, 'get'): history_c1.append(float(c1.get()))
    else: history_c1.append(float(c1))
    
    if hasattr(c2, 'get'): history_c2.append(float(c2.get()))
    else: history_c2.append(float(c2))
    
    # Metropolis Step
    beta = omega * 5.0 
    for _ in range(5): metro.step(beta)
    
    e_mh = metro.energy()
    if hasattr(e_mh, 'get'): history_mh.append(float(e_mh.get()))
    else: history_mh.append(float(e_mh))
    
    if i % 20 == 0:
        print(f"Step {i:3d} | Omega {omega:.2f} | SW Unsat: {unsat_sw:.4f} (C1={history_c1[-1]:.2f}) | MH Unsat: {history_mh[-1]:.4f}")

dt = time.time() - t0
print(f"Done in {dt:.2f}s")

# Plot
omega_cpu = omega_schedule
sw_cpu = np.array(history_sw)
c1_cpu = np.array(history_c1)
c2_cpu = np.array(history_c2)
mh_cpu = np.array(history_mh)

plt.figure(figsize=(12, 7))
ax1 = plt.gca()

# Energy Axis
l1, = ax1.plot(omega_cpu, sw_cpu, label='Stochastic SW Energy', color='cyan', linewidth=2)
l2, = ax1.plot(omega_cpu, mh_cpu, label='Metropolis Energy', color='orange', alpha=0.6)
ax1.set_xlabel(r'Coupling $\omega$')
ax1.set_ylabel('Fraction Unsatisfied', color='white')
ax1.tick_params(axis='y', labelcolor='white')
ax1.grid(True, alpha=0.2)

# Cluster Axis
ax2 = ax1.twinx()
l3, = ax2.plot(omega_cpu, c1_cpu, label='Largest Cluster (C1)', color='magenta', linestyle='--', linewidth=1.5)
l4, = ax2.plot(omega_cpu, c2_cpu, label='2nd Largest (C2)', color='lime', linestyle=':', linewidth=1.5)
ax2.set_ylabel('Cluster Size Fraction', color='white')
ax2.tick_params(axis='y', labelcolor='white')

# Legend
lines = [l1, l2, l3, l4]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title(f'Stochastic SW vs MH (N={N}, Alpha={alpha}) | Percolation')
plt.show()
"""
add_code(main_code)

with open("Swendsen-Wang_3SAT_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully.")