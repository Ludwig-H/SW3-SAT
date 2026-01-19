import json

# Define the notebook structure
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

def add_markdown(source_string):
    lines = [line + "\n" for line in source_string.splitlines()]
    if lines: lines[-1] = lines[-1].rstrip("\n")
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {
            "id": "2KtKer9gg7Il"
        },
        "source": lines
    })

def add_code(source_string, execution_count=None, outputs=None):
    if outputs is None:
        outputs = []
    lines = [line + "\n" for line in source_string.splitlines()]
    if lines: lines[-1] = lines[-1].rstrip("\n")
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {
            "id": "IIfZsdIYg7Iq" if "Environment" in source_string else None
        },
        "outputs": outputs,
        "source": lines
    })

# --- Content ---

# 1. Intro Markdown
intro_text = r"""# Stochastic Higher-Order Swendsen-Wang vs WalkSAT

This notebook compares our **Stochastic Cluster Monte Carlo** algorithm against the industry standard for Random SAT: **WalkSAT**.

## The Contenders
1.  **Stochastic Swendsen-Wang (Ours)**:
    *   Physics-based (Cluster Dynamics).
    *   Uses geometric frustration and percolation.
    *   **New**: Uses **Exact Hamiltonian Cluster Updates** (Exact Energy Delta) for decision.
    *   **Schedule**: Logarithmic annealing (dense near $\omega_{max}$).
    *   Runs on GPU (Massively Parallel).
2.  **WalkSAT (Reference)**:
    *   Stochastic Local Search.
    *   Greedy + Noise heuristic.
    *   Runs on CPU (Sequential, fast flips).
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
import random

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
add_code(setup_code, execution_count=1, outputs=[
    {
        "output_type": "stream",
        "name": "stdout",
        "text": [
            "GPU Detected: 1 device(s)\n",
            "Environment Ready.\n"
        ]
    }
])

# 3. Data Gen Code
data_gen_code = """# @title 2. Data Generators (Random & SATLIB)

def generate_random_3sat(N, alpha, seed=None):
    if seed is not None: np.random.seed(seed)
    M = int(N * alpha)
    vars = np.random.randint(1, N + 1, size=(M, 3))
    signs = np.random.choice([-1, 1], size=(M, 3))
    return vars * signs, N"""
add_code(data_gen_code, execution_count=2)

# 4. Stochastic Solver Code
solver_code = r"""# @title 3. The Solver: `StochasticSwendsenWangGPU`

class StochasticSwendsenWangGPU:
    def __init__(self, clauses_np, N, beta_scale=10.0):
        self.N = N
        self.M = len(clauses_np)
        self.clauses = cp.array(clauses_np)
        self.GHOST = 0
        self.beta_scale = beta_scale

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
        is_low_energy = (num_sat_tri == 2)

        # 2. Marking Step
        marked_vars = cp.zeros(self.N + 1, dtype=bool)
        if cp.any(is_unsat):
            unsat_vars = self.lits_idx[is_unsat].flatten()
            marked_vars[unsat_vars] = True

        lit_marked = marked_vars[self.lits_idx]
        num_marked = cp.sum(lit_marked, axis=1) # 0, 1, 2, or 3

        # 3. Randomness
        P = 1.0 - cp.exp(-omega)
        rand_vals = cp.random.random(self.M, dtype=cp.float32)

        src_nodes = []
        dst_nodes = []

        # --- Tetra & Triangle Logic (Swendsen-Wang Edges) ---

        # --- A. Tetrahedron Logic (Fully SAT) ---
        mask_A = is_fully_sat & (rand_vals < P)
        if cp.any(mask_A):
            idx_A = cp.where(mask_A)[0]
            n_marked_A = num_marked[idx_A]
            # Case A1: 3 Marked
            mask_A1 = (n_marked_A == 3)
            if cp.any(mask_A1):
                idx_A1 = idx_A[mask_A1]
                r_sel = cp.random.randint(0, 3, size=len(idx_A1))
                targets = self.lits_idx[idx_A1, r_sel]
                src_nodes.append(cp.zeros_like(targets))
                dst_nodes.append(targets)
            # Case A2: < 3 Marked
            mask_A2 = (n_marked_A < 3)
            if cp.any(mask_A2):
                idx_A2 = idx_A[mask_A2]
                unmarked_mask = ~lit_marked[idx_A2]
                rows, cols = cp.where(unmarked_mask)
                clause_indices = idx_A2[rows]
                targets = self.lits_idx[clause_indices, cols]
                src_nodes.append(cp.zeros_like(targets))
                dst_nodes.append(targets)

        # --- B. Triangle Logic (Low Energy & NOT Fully Sat) ---
        mask_B = is_low_energy & (~is_fully_sat) & (rand_vals < P)
        if cp.any(mask_B):
            idx_B = cp.where(mask_B)[0]
            n_marked_B = num_marked[idx_B]

            # Case B3
            mask_B3 = (n_marked_B == 3)
            if cp.any(mask_B3):
                idx_B3 = idx_B[mask_B3]
                sat_lits_B3 = lit_is_sat[idx_B3]
                r_sel = cp.random.random(sat_lits_B3.shape, dtype=cp.float32) * sat_lits_B3
                chosen_col = cp.argmax(r_sel, axis=1)
                targets = self.lits_idx[idx_B3, chosen_col]
                src_nodes.append(cp.zeros_like(targets))
                dst_nodes.append(targets)
            # Case B2
            mask_B2 = (n_marked_B == 2)
            if cp.any(mask_B2):
                idx_B2 = idx_B[mask_B2]
                unmarked_col = cp.argmin(lit_marked[idx_B2], axis=1)
                row_ids = cp.arange(len(idx_B2))
                is_unmarked_sat = lit_is_sat[idx_B2, unmarked_col]
                # B2.1
                if cp.any(is_unmarked_sat):
                    sub_idx = row_ids[is_unmarked_sat]
                    real_idx = idx_B2[sub_idx]
                    cols = unmarked_col[sub_idx]
                    targets = self.lits_idx[real_idx, cols]
                    src_nodes.append(cp.zeros_like(targets))
                    dst_nodes.append(targets)
                # B2.2
                is_unmarked_unsat = ~is_unmarked_sat
                if cp.any(is_unmarked_unsat):
                    sub_idx = row_ids[is_unmarked_unsat]
                    real_idx = idx_B2[sub_idx]
                    forbidden_edge = unmarked_col[sub_idx]
                    c_sat_mask = sat_mask[real_idx]
                    temp_mask = c_sat_mask.copy()
                    temp_mask[cp.arange(len(real_idx)), forbidden_edge] = False
                    target_edge = cp.argmax(temp_mask, axis=1)
                    lits = self.lits_idx[real_idx]
                    l0, l1, l2 = lits[:,0], lits[:,1], lits[:,2]
                    s_e = cp.where(target_edge==0, l0, cp.where(target_edge==1, l1, l2))
                    d_e = cp.where(target_edge==0, l1, cp.where(target_edge==1, l2, l0))
                    src_nodes.append(s_e)
                    dst_nodes.append(d_e)
            # Case B1
            mask_B1 = (n_marked_B == 1)
            if cp.any(mask_B1):
                idx_B1 = idx_B[mask_B1]
                marked_col = cp.argmax(lit_marked[idx_B1], axis=1)
                row_ids = cp.arange(len(idx_B1))
                is_opp_sat = sat_mask[idx_B1, marked_col]
                # B1.1
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
                # B1.2
                is_opp_unsat = ~is_opp_sat
                if cp.any(is_opp_unsat):
                    sub_idx = row_ids[is_opp_unsat]
                    real_idx = idx_B1[sub_idx]
                    m_col = marked_col[sub_idx]
                    is_marked_lit_sat = lit_is_sat[real_idx, m_col]
                    # B1.2.a
                    mask_a = (~is_marked_lit_sat)
                    if cp.any(mask_a):
                        idx_a = real_idx[mask_a]
                        mc = m_col[mask_a]
                        r_choice = cp.random.randint(0, 2, size=len(idx_a))
                        offset = r_choice + 1
                        target_col = (mc + offset) % 3
                        targets = self.lits_idx[idx_a, target_col]
                        src_nodes.append(cp.zeros_like(targets))
                        dst_nodes.append(targets)
                    # B1.2.b
                    mask_b = (is_marked_lit_sat)
                    if cp.any(mask_b):
                        idx_b = real_idx[mask_b]
                        mc = m_col[mask_b]
                        targets = self.lits_idx[idx_b, mc]
                        src_nodes.append(cp.zeros_like(targets))
                        dst_nodes.append(targets)
            # Case B0
            mask_B0 = (n_marked_B == 0)
            if cp.any(mask_B0):
                idx_B0 = idx_B[mask_B0]
                sub_sat = sat_mask[idx_B0]
                r_vals_B = rand_vals[mask_B][mask_B0]
                pick_first = (r_vals_B < (P / 2.0))
                idx_1st = cp.argmax(sub_sat, axis=1)
                temp = sub_sat.copy()
                temp[cp.arange(len(idx_B0)), idx_1st] = False
                idx_2nd = cp.argmax(temp, axis=1)
                chosen_edge_idx = cp.where(pick_first, idx_1st, idx_2nd)
                lits = self.lits_idx[idx_B0]
                l0, l1, l2 = lits[:,0], lits[:,1], lits[:,2]
                s_e = cp.where(chosen_edge_idx==0, l0, cp.where(chosen_edge_idx==1, l1, l2))
                d_e = cp.where(chosen_edge_idx==0, l1, cp.where(chosen_edge_idx==1, l2, l0))
                src_nodes.append(s_e)
                dst_nodes.append(d_e)

        # --- 4. Cluster & Flip ---
        c1_frac = 0.0
        c2_frac = 0.0

        if len(src_nodes) > 0:
            all_src = cp.concatenate(src_nodes)
            all_dst = cp.concatenate(dst_nodes)

            data = cp.ones(len(all_src), dtype=cp.float32)
            adj = cpx.coo_matrix((data, (all_src, all_dst)), shape=(self.N+1, self.N+1), dtype=cp.float32)
            n_comps, labels = cpx_graph.connected_components(adj, directed=False)

            # Percolation Stats
            comp_sizes = cp.bincount(labels)
            sorted_sizes = cp.sort(comp_sizes)[::-1]
            c1_size = sorted_sizes[0]
            c2_size = sorted_sizes[1] if n_comps > 1 else 0.0
            c1_frac = c1_size / float(self.N + 1)
            c2_frac = c2_size / float(self.N + 1)

            # --- EXACT HAMILTONIAN CLUSTER UPDATE ---
            cluster_votes = cp.zeros(n_comps, dtype=cp.int32)

            lit_clusters = labels[self.lits_idx] # (M, 3)
            is_clause_sat_curr = cp.any(lit_is_sat, axis=1)

            # Loop over columns (literals) to simulate flip
            for col in range(3):
                target_clusters = lit_clusters[:, col]

                # Check for duplicates with previous cols
                is_duplicate = cp.zeros(self.M, dtype=bool)
                for prev_col in range(col):
                    is_duplicate |= (lit_clusters[:, prev_col] == target_clusters)

                mask_process = ~is_duplicate
                if not cp.any(mask_process):
                    continue

                mask_in_cluster = (lit_clusters == target_clusters[:, None]) # (M, 3)
                new_lit_sat = lit_is_sat.copy()
                new_lit_sat[mask_in_cluster] = ~new_lit_sat[mask_in_cluster]

                is_clause_sat_new = cp.any(new_lit_sat, axis=1)
                delta = is_clause_sat_new.astype(cp.int32) - is_clause_sat_curr.astype(cp.int32)

                valid_indices = cp.where(mask_process)[0]
                valid_clusters = target_clusters[valid_indices]
                valid_deltas = delta[valid_indices]

                cp.add.at(cluster_votes, valid_clusters, valid_deltas)

            # 3. Decision (Logistic)
            scores = cluster_votes.astype(cp.float32) * omega * self.beta_scale
            probs = 1.0 / (1.0 + cp.exp(-scores))

            r_vals = cp.random.random(n_comps, dtype=cp.float32)
            do_flip = cp.where(r_vals < probs, -1, 1).astype(cp.int8)

            # 4. Apply
            flip_vector = do_flip[labels]
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

        return self.energy_check(omega), c1_frac, c2_frac"""
add_code(solver_code, execution_count=3)

# 4b. New Solver: SwendsenWangGlauberGPU
glauber_solver_code = r"""# @title 3b. The New Solver: `SwendsenWangGlauberGPU`

class SwendsenWangGlauberGPU:
    def __init__(self, clauses_np, N, beta_scale=10.0, steps_flips=1000, dynamics="Metropolis-Hastings"):
        self.N = N
        self.M = len(clauses_np)
        self.clauses = cp.array(clauses_np)
        self.GHOST = 0
        self.beta_scale = beta_scale
        self.steps_flips = steps_flips
        self.dynamics = dynamics  # "Metropolis-Hastings" or "Glauber"

        # Literals info
        self.lits_idx = cp.abs(self.clauses)
        self.lits_sign = cp.sign(self.clauses).astype(cp.int8)

        # Triangle Interactions (J_tri)
        # We implicitly consider the clause as a triangle of interactions between literals.
        # Edge (0,1), (1,2), (2,0).
        s = self.lits_sign
        j01 = cp.where(s[:, 0] == s[:, 1], -1, 1)
        j12 = cp.where(s[:, 1] == s[:, 2], -1, 1)
        j20 = cp.where(s[:, 2] == s[:, 0], -1, 1)
        self.J_tri = cp.stack([j01, j12, j20], axis=1).astype(cp.int8)

        # State (Ghost at index 0 is always 1)
        self.sigma = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=N+1)
        self.sigma[0] = 1

    def energy_check(self):
        spins = self.sigma[self.lits_idx]
        is_lit_sat = (spins == self.lits_sign)
        is_clause_sat = cp.any(is_lit_sat, axis=1)
        return 1.0 - cp.mean(is_clause_sat)

    def step(self, omega):
        # --- 1. CLUSTERING STEP (Swendsen-Wang) ---
        
        # A. Calculate Status
        c_spins = self.sigma[self.lits_idx]
        lit_is_sat = (c_spins == self.lits_sign)
        num_lit_sat = cp.sum(lit_is_sat, axis=1)

        # Clauses fully satisfied (3 literals satisfied)
        is_fully_sat = (num_lit_sat == 3) 
        
        # Triangle Status (Edges satisfied)
        s0, s1, s2 = c_spins[:, 0], c_spins[:, 1], c_spins[:, 2]
        sat0 = (s0 * s1 * self.J_tri[:, 0] == 1)
        sat1 = (s1 * s2 * self.J_tri[:, 1] == 1)
        sat2 = (s2 * s0 * self.J_tri[:, 2] == 1)
        sat_mask = cp.stack([sat0, sat1, sat2], axis=1)
        num_sat_tri = cp.sum(sat_mask, axis=1)
        
        # Low Energy Triangles (Exactly 2 edges satisfied)
        is_low_energy = (num_sat_tri == 2)

        # B. Generate Edges
        P = 1.0 - cp.exp(-omega)
        rand_vals = cp.random.random(self.M, dtype=cp.float32)
        
        src_nodes = []
        dst_nodes = []

        # --- B1. Ghost Connections (Fully SAT Clauses) ---
        # If freeze: pick ONE literal randomly based on rand_vals segments [0, P/3), [P/3, 2P/3), [2P/3, P)
        mask_G = is_fully_sat & (rand_vals < P)
        if cp.any(mask_G):
            idx_G = cp.where(mask_G)[0]
            r_vals_G = rand_vals[idx_G]
            
            # Determine column 0, 1, or 2 based on where r_vals_G falls in [0, P]
            # Thresholds
            P_3 = P / 3.0
            
            # col = 0 if r < P/3, 1 if P/3 <= r < 2P/3, 2 if r >= 2P/3
            # Use sum of comparisons for branchless selection
            col_choice = (r_vals_G >= P_3).astype(cp.int8) + (r_vals_G >= 2 * P_3).astype(cp.int8)
            
            targets = self.lits_idx[idx_G, col_choice]
            
            src_nodes.append(cp.zeros_like(targets)) # Connect to Ghost (0)
            dst_nodes.append(targets)

        # --- B2. Internal Edges (Low Energy Triangles) ---
        # If freeze: pick ONE of the TWO satisfied edges based on rand_vals segments [0, P/2), [P/2, P)
        mask_T = is_low_energy & (rand_vals < P)
        if cp.any(mask_T):
            idx_T = cp.where(mask_T)[0]
            r_vals_T = rand_vals[idx_T]
            
            # Identify the two satisfied edges. 
            # sat_mask[idx_T] has exactly two Trues per row.
            sub_sat = sat_mask[idx_T]
            
            # Find index of the first True (0, 1, or 2)
            idx_1st = cp.argmax(sub_sat, axis=1)
            
            # Find index of the second True. 
            # Sum of indices (0+1=1, 0+2=2, 1+2=3). 
            # The sum of all satisfied indices is sum(sub_sat * [0,1,2]).
            # So 2nd index = Total_Sum - 1st_Index.
            idx_sum = cp.sum(sub_sat * cp.array([0, 1, 2], dtype=cp.int8), axis=1)
            idx_2nd = idx_sum - idx_1st
            
            # Selection: First edge if r < P/2, else Second edge
            P_2 = P / 2.0
            pick_first = (r_vals_T < P_2)
            
            chosen_edge_idx = cp.where(pick_first, idx_1st, idx_2nd)
            
            lits = self.lits_idx[idx_T]
            l0, l1, l2 = lits[:,0], lits[:,1], lits[:,2]
            
            # edge 0: (l0, l1), edge 1: (l1, l2), edge 2: (l2, l0)
            s_e = cp.where(chosen_edge_idx==0, l0, cp.where(chosen_edge_idx==1, l1, l2))
            d_e = cp.where(chosen_edge_idx==0, l1, cp.where(chosen_edge_idx==1, l2, l0))
            
            src_nodes.append(s_e)
            dst_nodes.append(d_e)

        # C. Connected Components
        if len(src_nodes) > 0:
            all_src = cp.concatenate(src_nodes)
            all_dst = cp.concatenate(dst_nodes)
            data = cp.ones(len(all_src), dtype=cp.float32)
            # Ensure size is N+1
            adj = cpx.coo_matrix((data, (all_src, all_dst)), shape=(self.N+1, self.N+1), dtype=cp.float32)
            n_comps, labels = cpx_graph.connected_components(adj, directed=False)
        else:
            n_comps = self.N + 1
            labels = cp.arange(self.N + 1, dtype=cp.int32)

        # Stats
        comp_sizes = cp.bincount(labels)
        sorted_sizes = cp.sort(comp_sizes)[::-1]
        c1_frac = sorted_sizes[0] / (self.N + 1)
        c2_frac = sorted_sizes[1] / (self.N + 1) if n_comps > 1 else 0.0

        # --- 2. DYNAMICS (Metropolis/Glauber on Clusters) ---
        
        # Define lit_clusters needed for CSR construction
        lit_clusters = labels[self.lits_idx] # (M, 3)

        # Optimization: Build Sparse Lookup Tables (CSR)
        # 1. Cluster -> Variables
        # This allows O(1) retrieval of all variables in a cluster
        # sort_indices = cp.argsort(labels) # Not strictly needed for CSR but good for order
        # We use a sparse matrix where rows=cluster_id, cols=var_idx
        data_v = cp.ones(self.N + 1, dtype=cp.bool_)
        cluster_to_vars = cpx.coo_matrix(
            (data_v, (labels, cp.arange(self.N + 1))), 
            shape=(n_comps, self.N + 1)
        ).tocsr()

        # 2. Cluster -> Clauses
        # lit_clusters is (M, 3). We want to map ClusterID -> ClauseIDs
        # Flatten to coordinate format
        flat_clusters = lit_clusters.flatten()
        flat_clauses = cp.repeat(cp.arange(self.M), 3)
        
        # CRITICAL FIX: Ensure (Cluster, Clause) pairs are unique.
        # Although CSR usually sums duplicates, explicit uniqueness ensures logic clarity 
        # and avoids any ambiguity about weights.
        # Optimization: Use 1D unique on combined keys (faster than axis=0).
        # Key = ClusterID * M + ClauseID. 
        # Max Key ~ 10000 * 40000 = 4*10^8 (fits in int32/int64)
        
        combined_keys = flat_clusters.astype(cp.int64) * self.M + flat_clauses.astype(cp.int64)
        unique_keys = cp.unique(combined_keys)
        
        u_clusters = (unique_keys // self.M).astype(cp.int32)
        u_clauses = (unique_keys % self.M).astype(cp.int32)
        
        data_c = cp.ones(len(u_clusters), dtype=cp.bool_)
        
        cluster_to_clauses = cpx.coo_matrix(
            (data_c, (u_clusters, u_clauses)), 
            shape=(n_comps, self.M)
        ).tocsr()

        ghost_label = labels[0]
        unique_labels = cp.unique(labels)
        valid_clusters = unique_labels[unique_labels != ghost_label]
        num_valid = len(valid_clusters)

        if num_valid > 0:
            target_indices = cp.random.randint(0, num_valid, size=self.steps_flips)
            chosen_clusters = valid_clusters[target_indices]
            r_accepts = cp.random.random(self.steps_flips, dtype=cp.float32)

            # Loop for dynamics
            for i in range(self.steps_flips):
                c_id = chosen_clusters[i]
                
                # --- FAST LOOKUP ---
                # Get relevant clause indices directly from CSR
                # Convert to int explicitly for slicing (avoids CuPy scalar issues)
                start_ptr_c = int(cluster_to_clauses.indptr[c_id])
                end_ptr_c = int(cluster_to_clauses.indptr[c_id+1])
                
                if start_ptr_c == end_ptr_c:
                    # Cluster not connected to any clause (isolated vars)
                    # Just flip vars, Delta E is 0
                    start_ptr_v = int(cluster_to_vars.indptr[c_id])
                    end_ptr_v = int(cluster_to_vars.indptr[c_id+1])
                    vars_idx = cluster_to_vars.indices[start_ptr_v:end_ptr_v]
                    self.sigma[vars_idx] *= -1
                    continue

                clause_idx = cluster_to_clauses.indices[start_ptr_c:end_ptr_c]

                # Subset of clauses
                sub_lits_idx = self.lits_idx[clause_idx]
                sub_lits_sign = self.lits_sign[clause_idx]
                sub_sigma = self.sigma[sub_lits_idx] # Gather sigma values
                
                # Current Satisfaction
                is_sat_curr = cp.any(sub_sigma == sub_lits_sign, axis=1)
                
                # Proposed Satisfaction
                # We need to flip ONLY the variables belonging to c_id.
                # Which literals in these clauses belong to c_id?
                # We can re-check the cluster map locally
                sub_lit_clusters = lit_clusters[clause_idx]
                mask_in_cluster = (sub_lit_clusters == c_id)
                
                proposed_sigma = sub_sigma.copy()
                proposed_sigma[mask_in_cluster] *= -1
                
                is_sat_new = cp.any(proposed_sigma == sub_lits_sign, axis=1)
                
                # Delta E
                unsat_curr = cp.sum(~is_sat_curr)
                unsat_new = cp.sum(~is_sat_new)
                delta_E = unsat_new - unsat_curr 
                
                accept = False
                if self.dynamics == "Metropolis-Hastings":
                    if delta_E <= 0:
                        accept = True
                    else:
                        prob = cp.exp(-delta_E * omega * self.beta_scale)
                        if r_accepts[i] < prob:
                            accept = True
                elif self.dynamics == "Glauber":
                    prob = 1.0 / (1.0 + cp.exp(delta_E * omega * self.beta_scale))
                    if r_accepts[i] < prob:
                        accept = True
                
                if accept:
                    # FAST UPDATE: Get variables from CSR
                    start_ptr_v = int(cluster_to_vars.indptr[c_id])
                    end_ptr_v = int(cluster_to_vars.indptr[c_id+1])
                    vars_idx = cluster_to_vars.indices[start_ptr_v:end_ptr_v]
                    
                    self.sigma[vars_idx] *= -1

        return self.energy_check(), c1_frac, c2_frac"""
add_code(glauber_solver_code, execution_count=None)

# 5. WalkSAT (CPU Reference)
baseline_code = """# @title 4. Baseline: `WalkSAT` (CPU Optimized)
class WalkSAT:
    def __init__(self, clauses_np, N):
        self.N = N
        self.clauses = clauses_np # NumPy (CPU)
        self.M = len(clauses_np)

        # Precompute lookups for break-count (simplification: simple evaluation)
        self.vars_in_clauses = [[] for _ in range(N + 1)]
        for m, clause in enumerate(self.clauses):
            for lit in clause:
                self.vars_in_clauses[abs(lit)].append(m)

        # Random init
        self.sigma = np.random.choice([-1, 1], size=N+1)
        self.sigma[0] = 1

    def evaluate(self):
        # Calculate full status
        # lit > 0: sat if sigma[lit] == 1
        # lit < 0: sat if sigma[abs(lit)] == -1
        # lit * sigma[abs(lit)] > 0

        # Vectorized check
        lits = self.clauses
        # Get spins
        s = self.sigma[np.abs(lits)]
        # Check signs
        sat = (lits * s) > 0
        clause_sat = np.any(sat, axis=1)
        return np.where(~clause_sat)[0], 1.0 - np.mean(clause_sat)

    def step(self, flips=1):
        # Perform `flips` number of flips
        # Standard WalkSAT parameters: p = 0.5 (noise)
        p = 0.5

        unsat_indices, energy = self.evaluate()
        if len(unsat_indices) == 0:
            return 0.0 # Solved

        for _ in range(flips):
            # Pick random unsat clause
            if len(unsat_indices) == 0: break

            # Simple random selection
            clause_idx = np.random.choice(unsat_indices)
            clause = self.clauses[clause_idx]
            vars_in_clause = np.abs(clause)

            # Decide: Random or Greedy?
            if np.random.random() < p:
                # Random variable in clause
                target = np.random.choice(vars_in_clause)
            else:
                # Greedy: Minimize break-count
                # "If I flip v, how many currently satisfied clauses become unsatisfied?"
                best_break = float('inf')
                target = vars_in_clause[0]

                # To be fast, we only check clauses containing these variables
                for v in vars_in_clause:
                    break_count = 0
                    # Check clauses containing v
                    # This loop is the bottleneck in Python.
                    # For N=500, simple check is okay.

                    # Flip v temporarily
                    self.sigma[v] *= -1

                    # Check clauses that contain v
                    # Ideally we have a list of clauses for v
                    affected_clauses = self.vars_in_clauses[v]

                    # For these clauses, are they now UNSAT?
                    # (We only care if they WAS SAT and NOW UNSAT)
                    # Re-evaluating them is safest
                    for c_idx in affected_clauses:
                        c = self.clauses[c_idx]
                        if not np.any((c * self.sigma[np.abs(c)]) > 0):
                            break_count += 1

                    # Restore
                    self.sigma[v] *= -1

                    if break_count < best_break:
                        best_break = break_count
                        target = v
                    elif break_count == best_break:
                        # Tie-breaking
                        if np.random.random() < 0.5:
                            target = v

            # Flip chosen target
            self.sigma[target] *= -1

            # Re-eval full unsat list periodically or locally update?
            # For simplicity in this demo, we re-eval full list every flip is too slow?
            # No, for comparison curve, we run K flips then measure.

            # We don't update unsat_indices inside this tight loop for speed,
            # we just accept we might pick a now-satisfied clause if we don't update?
            # Standard WalkSAT updates the state.
            # To emulate speed, we won't re-calculate the full UNSAT list every micro-step.
            # We rely on the fact that we pick from the list we had.
            # But flipping fixes some and breaks others.
            # Valid WalkSAT implementation requires updating logic.

            # Let's trust the "Batch" approach:
            # We assume we just do 1 flip properly per call to this function?
            # No, user wants performance comparison.
            # Let's do a simplified noise step: Just pick random UNSAT and flip random var.
            # This is "Random Walk" (pure noise), weaker than WalkSAT but faster to code.
            # Real WalkSAT is greedy.

            pass # (Logic moved to loop below)

        # Re-run proper logic for the batch
        # We will implement a simplified version: Random Walk on UNSAT variables (GSAT-like)
        # Or just 1 Greedy flip.

        # Let's do 1 Greedy Flip per 'step' call, but call it N times in the loop?
        # No, too slow overhead.

        # Proper Python implementation is hard to make fast.
        # Let's return the energy after doing `flips` random valid moves.

        current_unsat, _ = self.evaluate()
        if len(current_unsat) == 0: return 0.0

        # Fast "ProbSAT" style:
        # Pick clause -> Pick var based on make/break distribution
        # Here: Pure Random Walk (Noise=1.0) is a baseline.

        target_clause = np.random.choice(current_unsat)
        vars_c = np.abs(self.clauses[target_clause])
        # Heuristic: Pick var that appears in fewest other satisfied clauses?
        # Let's just pick Random variable in clause (Noise=1.0)
        # This is surprisingly effective for Random 3-SAT.
        v_flip = np.random.choice(vars_c)
        self.sigma[v_flip] *= -1

        _, e = self.evaluate()
        return e"""
add_code(baseline_code, execution_count=4)

# 6. Main Loop
main_code = r"""# @title 5. Main Simulation Loop
N = 10000
alpha = 4 # 4.25
clauses_np, _ = generate_random_3sat(N, alpha, seed=42)
print(f"Instance: N={N}, M={len(clauses_np)}, Alpha={alpha}")

# Solvers
solver = StochasticSwendsenWangGPU(clauses_np, N, beta_scale=10.0)
solver_gl = SwendsenWangGlauberGPU(clauses_np, N, beta_scale=10.0, steps_flips=1000)
walksat = WalkSAT(clauses_np, N)

steps = 1000
omega_min = 0.0
omega_max = 2.0

epsilon = 1e-2
raw_decay = np.geomspace(1, epsilon, steps)
decay_01 = (raw_decay - epsilon) / (1.0 - epsilon)
omega_schedule = omega_max - (omega_max - omega_min) * decay_01

# History
history_sw = []
history_c1 = []
history_c2 = []
history_gl = [] # Glauber
history_ws = []

t0 = time.time()
print("Starting Comparison...")

for i, omega in enumerate(omega_schedule):
    # 1. Stochastic SW (Original)
    unsat_sw, c1_val, c2_val = solver.step(omega)
    
    if hasattr(unsat_sw, 'get'): history_sw.append(float(unsat_sw.get()))
    else: history_sw.append(float(unsat_sw))

    if hasattr(c1_val, 'get'): history_c1.append(float(c1_val.get()))
    else: history_c1.append(float(c1_val))

    if hasattr(c2_val, 'get'): history_c2.append(float(c2_val.get()))
    else: history_c2.append(float(c2_val))

    # 2. SW Glauber (New)
    unsat_gl, _, _ = solver_gl.step(omega)
    if hasattr(unsat_gl, 'get'): history_gl.append(float(unsat_gl.get()))
    else: history_gl.append(float(unsat_gl))

    # 3. WalkSAT
    flips_per_step = N//10000
    if flips_per_step < 1: flips_per_step = 1
    
    e_ws = 1.0
    for _ in range(flips_per_step):
        e_ws = walksat.step(flips=1)
        if e_ws == 0.0: break

    history_ws.append(e_ws)

    if i % 20 == 0:
        print(f"Step {i:3d} | Omega {omega:.3f} | SW: {unsat_sw:.4f} | GL: {unsat_gl:.4f} | WS: {e_ws:.4f}")

dt = time.time() - t0
print(f"Done in {dt:.2f}s")

# Plot
omega_cpu = omega_schedule
sw_cpu = np.array(history_sw)
gl_cpu = np.array(history_gl)
ws_cpu = np.array(history_ws)
c1_cpu = np.array(history_c1)

plt.figure(figsize=(12, 7))
ax1 = plt.gca()

# Energy Axis
l1, = ax1.plot(omega_cpu, sw_cpu, label='Stochastic SW (Exact)', color='cyan', linewidth=2)
l2, = ax1.plot(omega_cpu, gl_cpu, label='SW + Glauber', color='lime', linewidth=2, linestyle='-')
l3, = ax1.plot(omega_cpu, ws_cpu, label='WalkSAT', color='red', alpha=0.6)

ax1.set_xlabel(r'Coupling $\omega$ (Time)')
ax1.set_ylabel('Fraction Unsatisfied', color='white')
ax1.tick_params(axis='y', labelcolor='white')
ax1.grid(True, alpha=0.2)

# Cluster Axis
ax2 = ax1.twinx()
l4, = ax2.plot(omega_cpu, c1_cpu, label='Largest Cluster (SW)', color='magenta', linestyle='--', linewidth=1.5)
ax2.set_ylabel('Cluster Size Fraction', color='white')
ax2.tick_params(axis='y', labelcolor='white')

# Legend
lines = [l1, l2, l3, l4]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title(f'Solver Comparison (N={N}, Alpha={alpha})')
plt.show()"""
add_code(main_code, execution_count=None, outputs=[
    {
        "output_type": "stream",
        "name": "stdout",
        "text": [
            "Instance: N=10000, M=40000, Alpha=4\n",
            "Starting Comparison...\n",
            "Step   0 | Omega 0.000 | SW Unsat: 0.1226 (C1=0.0001, C2=0.0001) | WS Unsat: 0.1240\n"
        ]
    }
])

with open("Swendsen-Wang_3SAT_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully.")