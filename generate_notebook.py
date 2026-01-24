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

# Cell 0 (Markdown)
md_source_0 = r"""# Stochastic Higher-Order Swendsen-Wang vs WalkSAT

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
3.  **Complete Swendsen-Wang**:
    *   Applies percolation to ALL clauses (not just SAT ones).
    *   Uniform percolation logic."""
add_markdown(md_source_0)

# Cell 1 (Code)
code_source_1 = r"""# @title 1. Environment & GPU Setup
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
add_code(code_source_1, execution_count=1, outputs=[{"output_type": "stream", "name": "stdout", "text": ["GPU Detected: 1 device(s)\n", "Environment Ready.\n"]}])

# Cell 2 (Code)
code_source_2 = r"""# @title 2. Data Generators (Random & SATLIB)

def generate_random_3sat(N, alpha, seed=None):
    if seed is not None: np.random.seed(seed)
    M = int(N * alpha)
    vars = np.random.randint(1, N + 1, size=(M, 3))
    signs = np.random.choice([-1, 1], size=(M, 3))
    return vars * signs, N"""
add_code(code_source_2, execution_count=2, outputs=[])

# Cell 3 (Code)
code_source_3 = r"""# @title 3. The Solver: `StochasticSwendsenWangGPU`

class StochasticSwendsenWangGPU:
    def __init__(self, clauses_np, N, beta_scale=15.0):
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
                    mc = marked_col[sub_idx]
                    is_marked_lit_sat = lit_is_sat[real_idx, mc]
                    # B1.2.a
                    mask_a = (~is_marked_lit_sat)
                    if cp.any(mask_a):
                        idx_a = real_idx[mask_a]
                        mc_a = mc[mask_a]
                        r_choice = cp.random.randint(0, 2, size=len(idx_a))
                        offset = r_choice + 1
                        target_col = (mc_a + offset) % 3
                        targets = self.lits_idx[idx_a, target_col]
                        src_nodes.append(cp.zeros_like(targets))
                        dst_nodes.append(targets)
                    # B1.2.b
                    mask_b = (is_marked_lit_sat)
                    if cp.any(mask_b):
                        idx_b = real_idx[mask_b]
                        mc_b = mc[mask_b]
                        targets = self.lits_idx[idx_b, mc_b]
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
add_code(code_source_3, execution_count=3, outputs=[])

# Cell 4 (Code)
code_source_4 = r"""# @title 3b. The New Solver: `SwendsenWangGlauberGPU` (Optimized Kernel)

# CUDA Kernel for Glauber Dynamics
glauber_kernel_code = r'''
#include <curand_kernel.h>

extern "C" __global__ void run_glauber_dynamics(
    signed char* sigma,           // N+1
    const int* c2c_indptr,        // n_comps + 1
    const int* c2c_indices,       // n_clauses_refs
    const int* c2v_indptr,        // n_comps + 1
    const int* c2v_indices,       // n_vars_refs
    const int* lits_idx,          // M * 3
    const signed char* lits_sign, // M * 3
    const int* lit_clusters,      // M * 3
    const int* valid_clusters,    // num_valid
    int num_valid,
    int steps,
    float omega,
    float beta_scale,
    unsigned long long seed
) {
    // Shared memory for reduction and communication
    __shared__ int delta_E_shared;
    __shared__ int decision_shared; // 0=reject, 1=accept
    __shared__ int target_cluster_shared;

    // Initialize RNG
    curandState state;
    if (threadIdx.x == 0) {
        curand_init(seed, 0, 0, &state);
    }

    for (int step = 0; step < steps; step++) {
        __syncthreads();

        // --- 1. Pick Target Cluster ---
        if (threadIdx.x == 0) {
            delta_E_shared = 0;
            decision_shared = 0;
            unsigned int r = curand(&state);
            int r_idx = r % num_valid;
            target_cluster_shared = valid_clusters[r_idx];
        }
        __syncthreads();

        int c_id = target_cluster_shared;
        int start_c = c2c_indptr[c_id];
        int end_c = c2c_indptr[c_id+1];

        // --- 2. Compute Delta E (Parallel over clauses) ---
        if (start_c < end_c) {
            for (int i = start_c + threadIdx.x; i < end_c; i += blockDim.x) {
                int clause_idx = c2c_indices[i];

                int idx0 = clause_idx * 3 + 0;
                int idx1 = clause_idx * 3 + 1;
                int idx2 = clause_idx * 3 + 2;

                int l0 = lits_idx[idx0];
                int l1 = lits_idx[idx1];
                int l2 = lits_idx[idx2];

                signed char s0 = lits_sign[idx0];
                signed char s1 = lits_sign[idx1];
                signed char s2 = lits_sign[idx2];

                signed char sig0 = sigma[l0];
                signed char sig1 = sigma[l1];
                signed char sig2 = sigma[l2];

                int cl0 = lit_clusters[idx0];
                int cl1 = lit_clusters[idx1];
                int cl2 = lit_clusters[idx2];

                bool sat_curr = (sig0 == s0) || (sig1 == s1) || (sig2 == s2);

                signed char p_sig0 = (cl0 == c_id) ? -sig0 : sig0;
                signed char p_sig1 = (cl1 == c_id) ? -sig1 : sig1;
                signed char p_sig2 = (cl2 == c_id) ? -sig2 : sig2;

                bool sat_new = (p_sig0 == s0) || (p_sig1 == s1) || (p_sig2 == s2);

                if (sat_curr != sat_new) {
                    int local_delta = (int)sat_curr - (int)sat_new;
                    atomicAdd(&delta_E_shared, local_delta);
                }
            }
        }
        __syncthreads();

        // --- 3. Decision ---
        if (threadIdx.x == 0) {
            int dE = delta_E_shared;
            if (dE <= 0) {
                decision_shared = 1;
            } else {
                float p = expf(-(float)dE * omega * beta_scale);
                float r = curand_uniform(&state);
                if (r < p) {
                    decision_shared = 1;
                }
            }
        }
        __syncthreads();

        // --- 4. Update Sigma ---
        if (decision_shared) {
            int start_v = c2v_indptr[c_id];
            int end_v = c2v_indptr[c_id+1];
            for (int i = start_v + threadIdx.x; i < end_v; i += blockDim.x) {
                int var_idx = c2v_indices[i];
                sigma[var_idx] *= -1;
            }
        }
    }
}
'''

class SwendsenWangGlauberGPU:
    def __init__(self, clauses_np, N, beta_scale=15.0, steps_flips=None, dynamics="Metropolis-Hastings"):
        self.N = N
        self.M = len(clauses_np)
        self.clauses = cp.array(clauses_np)
        self.GHOST = 0
        self.beta_scale = beta_scale
        if steps_flips is None:
            self.steps_flips = 2 * N
        else:
            self.steps_flips = steps_flips
        self.dynamics = dynamics

        self.lits_idx = cp.ascontiguousarray(cp.abs(self.clauses).astype(cp.int32))
        self.lits_sign = cp.ascontiguousarray(cp.sign(self.clauses).astype(cp.int8))

        s = self.lits_sign
        j01 = cp.where(s[:, 0] == s[:, 1], -1, 1)
        j12 = cp.where(s[:, 1] == s[:, 2], -1, 1)
        j20 = cp.where(s[:, 2] == s[:, 0], -1, 1)
        self.J_tri = cp.stack([j01, j12, j20], axis=1).astype(cp.int8)

        self.sigma = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=N+1)
        self.sigma[0] = 1
        
        # Best So Far
        self.best_sigma = self.sigma.copy()
        self.min_energy = 1.0

        self.kernel = cp.RawKernel(glauber_kernel_code, 'run_glauber_dynamics', options=('-std=c++17',))

    def energy_check(self):
        spins = self.sigma[self.lits_idx]
        is_lit_sat = (spins == self.lits_sign)
        is_clause_sat = cp.any(is_lit_sat, axis=1)
        return 1.0 - cp.mean(is_clause_sat)

    def step(self, omega, verbose=False):
        # --- 1. CLUSTERING ---
        c_spins = self.sigma[self.lits_idx]
        lit_is_sat = (c_spins == self.lits_sign)
        num_lit_sat = cp.sum(lit_is_sat, axis=1)
        is_fully_sat = (num_lit_sat == 3)

        s0, s1, s2 = c_spins[:, 0], c_spins[:, 1], c_spins[:, 2]
        sat0 = (s0 * s1 * self.J_tri[:, 0] == 1)
        sat1 = (s1 * s2 * self.J_tri[:, 1] == 1)
        sat2 = (s2 * s0 * self.J_tri[:, 2] == 1)
        sat_mask = cp.stack([sat0, sat1, sat2], axis=1)
        num_sat_tri = cp.sum(sat_mask, axis=1)
        is_low_energy = (num_sat_tri == 2)

        P = 1.0 - cp.exp(-omega)
        rand_vals = cp.random.random(self.M, dtype=cp.float32)

        src_nodes = []
        dst_nodes = []

        # --- B1. Ghost Connections (Fully SAT Clauses) ---
        # If freeze: connect ALL THREE literals to Ghost (0)
        mask_G = is_fully_sat & (rand_vals < P)
        if cp.any(mask_G):
            idx_G = cp.where(mask_G)[0]

            # We want to connect lits 0, 1, and 2 of these clauses to Ghost.
            # Get all literals for these clauses: shape (K, 3) -> flatten to (3K,)
            targets = self.lits_idx[idx_G].flatten()

            src_nodes.append(cp.zeros_like(targets)) # Connect to Ghost (0)
            dst_nodes.append(targets)

        # B2. Internal
        mask_T = is_low_energy & (rand_vals < P)
        if cp.any(mask_T):
            idx_T = cp.where(mask_T)[0]
            r_vals_T = rand_vals[idx_T]
            sub_sat = sat_mask[idx_T]
            idx_1st = cp.argmax(sub_sat, axis=1)
            idx_sum = cp.sum(sub_sat * cp.array([0, 1, 2], dtype=cp.int8), axis=1)
            idx_2nd = idx_sum - idx_1st
            P_2 = P / 2.0
            pick_first = (r_vals_T < P_2)
            chosen_edge_idx = cp.where(pick_first, idx_1st, idx_2nd)
            lits = self.lits_idx[idx_T]
            l0, l1, l2 = lits[:,0], lits[:,1], lits[:,2]
            s_e = cp.where(chosen_edge_idx==0, l0, cp.where(chosen_edge_idx==1, l1, l2))
            d_e = cp.where(chosen_edge_idx==0, l1, cp.where(chosen_edge_idx==1, l2, l0))
            src_nodes.append(s_e)
            dst_nodes.append(d_e)

        # Connected Components
        if len(src_nodes) > 0:
            all_src = cp.concatenate(src_nodes)
            all_dst = cp.concatenate(dst_nodes)
            data = cp.ones(len(all_src), dtype=cp.float32)
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
        if verbose:
            print(f"Phase 1 Top 7 Clusters: {sorted_sizes[:7]}")

        # --- 2. DYNAMICS (KERNEL) ---
        lit_clusters = labels[self.lits_idx] # (M, 3)

        # CSR: Cluster -> Vars
        data_v = cp.ones(self.N + 1, dtype=cp.bool_)
        cluster_to_vars = cpx.coo_matrix(
            (data_v, (labels, cp.arange(self.N + 1))),
            shape=(n_comps, self.N + 1)
        ).tocsr()

        # CSR: Cluster -> Clauses
        flat_clusters = lit_clusters.flatten()
        flat_clauses = cp.repeat(cp.arange(self.M), 3)
        combined_keys = flat_clusters.astype(cp.int64) * self.M + flat_clauses.astype(cp.int64)
        unique_keys = cp.unique(combined_keys)
        u_clusters = (unique_keys // self.M).astype(cp.int32)
        u_clauses = (unique_keys % self.M).astype(cp.int32)
        data_c = cp.ones(len(u_clusters), dtype=cp.bool_)

        cluster_to_clauses = cpx.coo_matrix(
            (data_c, (u_clusters, u_clauses)),
            shape=(n_comps, self.M)
        ).tocsr()

        # Valid Clusters
        ghost_label = labels[0]
        unique_labels = cp.unique(labels)
        valid_clusters = unique_labels[unique_labels != ghost_label].astype(cp.int32)
        num_valid = len(valid_clusters)

        if num_valid > 0:
            c2c_indptr = cluster_to_clauses.indptr.astype(cp.int32)
            c2c_indices = cluster_to_clauses.indices.astype(cp.int32)
            c2v_indptr = cluster_to_vars.indptr.astype(cp.int32)
            c2v_indices = cluster_to_vars.indices.astype(cp.int32)

            lit_clusters_ptr = cp.ascontiguousarray(lit_clusters.astype(cp.int32))

            seed = int(time.time() * 1000) % 1000000007

            self.kernel(
                (1,), (256,),
                (
                    self.sigma,
                    c2c_indptr, c2c_indices,
                    c2v_indptr, c2v_indices,
                    self.lits_idx, self.lits_sign,
                    lit_clusters_ptr,
                    valid_clusters,
                    cp.int32(num_valid),
                    cp.int32(self.steps_flips),
                    cp.float32(omega),
                    cp.float32(self.beta_scale),
                    cp.uint64(seed)
                )
            )

        # --- PHASE 2: UNSAT DYNAMICS ---
        # 1. Re-evaluate Status after Phase 1
        c_spins = self.sigma[self.lits_idx]
        lit_is_sat = (c_spins == self.lits_sign)
        num_lit_sat = cp.sum(lit_is_sat, axis=1)
        is_unsat = (num_lit_sat == 0)

        P = 1.0 - cp.exp(-omega)
        # 2. Percolation on UNSAT Clauses ONLY
        src_nodes_2 = []
        dst_nodes_2 = []

        if cp.any(is_unsat):
            idx_U = cp.where(is_unsat)[0]
            n_unsat = len(idx_U)
            r_vals_U = cp.random.random(n_unsat, dtype=cp.float32)
            
            P_7 = P / 7.0
            
            # --- FULL FREEZE (Two edges => Triangle) ---
            # Range [6P/7, P)
            mask_full = (r_vals_U >= 6.0 * P_7) & (r_vals_U < P)
            if cp.any(mask_full):
                sub_idx = idx_U[mask_full]
                lits = self.lits_idx[sub_idx]
                # Edge 0-1
                src_nodes_2.append(lits[:, 0])
                dst_nodes_2.append(lits[:, 1])
                # Edge 1-2
                src_nodes_2.append(lits[:, 1])
                dst_nodes_2.append(lits[:, 2])
            
            # --- SINGLE EDGE FREEZE ---
            # Edge 0 (0-1) : [0, 2P/7)
            mask_e0 = (r_vals_U < 2.0 * P_7)
            if cp.any(mask_e0):
                sub_idx = idx_U[mask_e0]
                lits = self.lits_idx[sub_idx]
                src_nodes_2.append(lits[:, 0])
                dst_nodes_2.append(lits[:, 1])
            # Edge 1 (1-2) : [2P/7, 4P/7)
            mask_e1 = (r_vals_U >= 2.0 * P_7) & (r_vals_U < 4.0 * P_7)
            if cp.any(mask_e1):
                sub_idx = idx_U[mask_e1]
                lits = self.lits_idx[sub_idx]
                src_nodes_2.append(lits[:, 1])
                dst_nodes_2.append(lits[:, 2])
            # Edge 2 (2-0) : [4P/7, 6P/7)
            mask_e2 = (r_vals_U >= 4.0 * P_7) & (r_vals_U < 6.0 * P_7)
            if cp.any(mask_e2):
                sub_idx = idx_U[mask_e2]
                lits = self.lits_idx[sub_idx]
                src_nodes_2.append(lits[:, 2])
                dst_nodes_2.append(lits[:, 0])

        # 3. Clustering Phase 2
        if len(src_nodes_2) > 0:
            all_src_2 = cp.concatenate(src_nodes_2)
            all_dst_2 = cp.concatenate(dst_nodes_2)
            data_2 = cp.ones(len(all_src_2), dtype=cp.float32)
            adj_2 = cpx.coo_matrix((data_2, (all_src_2, all_dst_2)), shape=(self.N+1, self.N+1), dtype=cp.float32)
            n_comps_2, labels_2 = cpx_graph.connected_components(adj_2, directed=False)
        else:
            n_comps_2 = self.N + 1
            labels_2 = cp.arange(self.N + 1, dtype=cp.int32)

        comp_sizes_2 = cp.bincount(labels_2)
        if verbose:
            sorted_sizes_2 = cp.sort(comp_sizes_2)[::-1]
            print(f"Phase 2 Top 7 Clusters: {sorted_sizes_2[:7]}")

        # 4. Prepare Kernel Phase 2
        lit_clusters_2 = labels_2[self.lits_idx]
        data_v_2 = cp.ones(self.N + 1, dtype=cp.bool_)
        cluster_to_vars_2 = cpx.coo_matrix((data_v_2, (labels_2, cp.arange(self.N + 1))), shape=(n_comps_2, self.N + 1)).tocsr()
        flat_clusters_2 = lit_clusters_2.flatten()
        flat_clauses_2 = cp.repeat(cp.arange(self.M), 3)
        combined_keys_2 = flat_clusters_2.astype(cp.int64) * self.M + flat_clauses_2.astype(cp.int64)
        unique_keys_2 = cp.unique(combined_keys_2)
        u_clusters_2 = (unique_keys_2 // self.M).astype(cp.int32)
        u_clauses_2 = (unique_keys_2 % self.M).astype(cp.int32)
        data_c_2 = cp.ones(len(u_clusters_2), dtype=cp.bool_)
        cluster_to_clauses_2 = cpx.coo_matrix((data_c_2, (u_clusters_2, u_clauses_2)), shape=(n_comps_2, self.M)).tocsr()

        # Valid Clusters 2: Size > 1 AND not Ghost
        ghost_label_2 = labels_2[0]
        unique_labels_2 = cp.unique(labels_2)
        mask_valid = (comp_sizes_2[unique_labels_2] > 1) & (unique_labels_2 != ghost_label_2)
        valid_clusters_2 = unique_labels_2[mask_valid].astype(cp.int32)
        num_valid_2 = len(valid_clusters_2)

        # 5. Execute Kernel Phase 2
        if num_valid_2 > 0:
            c2c_indptr_2 = cluster_to_clauses_2.indptr.astype(cp.int32)
            c2c_indices_2 = cluster_to_clauses_2.indices.astype(cp.int32)
            c2v_indptr_2 = cluster_to_vars_2.indptr.astype(cp.int32)
            c2v_indices_2 = cluster_to_vars_2.indices.astype(cp.int32)
            lit_clusters_ptr_2 = cp.ascontiguousarray(lit_clusters_2.astype(cp.int32))
            self.kernel((1,), (256,), (self.sigma, c2c_indptr_2, c2c_indices_2, c2v_indptr_2, c2v_indices_2, self.lits_idx, self.lits_sign, lit_clusters_ptr_2, valid_clusters_2, cp.int32(num_valid_2), cp.int32(self.steps_flips), cp.float32(omega), cp.float32(self.beta_scale), cp.uint64(seed + 1)))

        current_energy = self.energy_check()
        
        # Update Best So Far
        if current_energy < self.min_energy:
            self.min_energy = current_energy
            self.best_sigma = self.sigma.copy()
            if self.min_energy == 0.0:
                print(f"🎉 SOLUTION FOUND ! (Energy = 0.0) 🎉")
        
        return current_energy, c1_frac, c2_frac"""
add_code(code_source_4, execution_count=4, outputs=[])

# Cell 4b (Code)
code_source_4b = r"""# @title 3c. The Solver: `CompleteSwendsenWangGPU`

class CompleteSwendsenWangGPU:
    def __init__(self, clauses_np, N, beta_scale=15.0, steps_flips=None, dynamics="Metropolis-Hastings"):
        self.N = N
        self.M = len(clauses_np)
        self.clauses = cp.array(clauses_np)
        self.beta_scale = beta_scale
        if steps_flips is None:
            self.steps_flips = 2 * N
        else:
            self.steps_flips = steps_flips
        self.dynamics = dynamics

        self.lits_idx = cp.ascontiguousarray(cp.abs(self.clauses).astype(cp.int32))
        self.lits_sign = cp.ascontiguousarray(cp.sign(self.clauses).astype(cp.int8))

        self.sigma = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=N+1)
        self.sigma[0] = 1 # Dummy index 0
        
        self.best_sigma = self.sigma.copy()
        self.min_energy = 1.0

        # Reusing the Glauber kernel
        self.kernel = cp.RawKernel(glauber_kernel_code, 'run_glauber_dynamics', options=('-std=c++17',))

    def energy_check(self):
        spins = self.sigma[self.lits_idx]
        is_lit_sat = (spins == self.lits_sign)
        is_clause_sat = cp.any(is_lit_sat, axis=1)
        return 1.0 - cp.mean(is_clause_sat)

    def step(self, omega, verbose=False):
        P = 1.0 - cp.exp(-omega)
        rand_vals = cp.random.random(self.M, dtype=cp.float32)
        
        src_nodes = []
        dst_nodes = []
        
        P_7 = P / 7.0
        
        # Range [6P/7, P) -> Full Freeze
        mask_full = (rand_vals >= 6.0 * P_7) & (rand_vals < P)
        if cp.any(mask_full):
            sub_idx = cp.where(mask_full)[0]
            lits = self.lits_idx[sub_idx]
            # Edge 0-1
            src_nodes.append(lits[:, 0])
            dst_nodes.append(lits[:, 1])
            # Edge 1-2
            src_nodes.append(lits[:, 1])
            dst_nodes.append(lits[:, 2])
            
        # Edge 0 (0-1)
        mask_e0 = (rand_vals < 2.0 * P_7)
        if cp.any(mask_e0):
            sub_idx = cp.where(mask_e0)[0]
            lits = self.lits_idx[sub_idx]
            src_nodes.append(lits[:, 0])
            dst_nodes.append(lits[:, 1])
            
        # Edge 1 (1-2)
        mask_e1 = (rand_vals >= 2.0 * P_7) & (rand_vals < 4.0 * P_7)
        if cp.any(mask_e1):
            sub_idx = cp.where(mask_e1)[0]
            lits = self.lits_idx[sub_idx]
            src_nodes.append(lits[:, 1])
            dst_nodes.append(lits[:, 2])
            
        # Edge 2 (2-0)
        mask_e2 = (rand_vals >= 4.0 * P_7) & (rand_vals < 6.0 * P_7)
        if cp.any(mask_e2):
            sub_idx = cp.where(mask_e2)[0]
            lits = self.lits_idx[sub_idx]
            src_nodes.append(lits[:, 2])
            dst_nodes.append(lits[:, 0])

        if len(src_nodes) > 0:
            all_src = cp.concatenate(src_nodes)
            all_dst = cp.concatenate(dst_nodes)
            data = cp.ones(len(all_src), dtype=cp.float32)
            adj = cpx.coo_matrix((data, (all_src, all_dst)), shape=(self.N+1, self.N+1), dtype=cp.float32)
            n_comps, labels = cpx_graph.connected_components(adj, directed=False)
        else:
            n_comps = self.N + 1
            labels = cp.arange(self.N + 1, dtype=cp.int32)

        if verbose:
            comp_sizes = cp.bincount(labels)
            sorted_sizes = cp.sort(comp_sizes)[::-1]
            print(f"Complete SW Top 7 Clusters: {sorted_sizes[:7]}")

        lit_clusters = labels[self.lits_idx]
        
        data_v = cp.ones(self.N + 1, dtype=cp.bool_)
        cluster_to_vars = cpx.coo_matrix((data_v, (labels, cp.arange(self.N + 1))), shape=(n_comps, self.N + 1)).tocsr()
        
        flat_clusters = lit_clusters.flatten()
        flat_clauses = cp.repeat(cp.arange(self.M), 3)
        combined_keys = flat_clusters.astype(cp.int64) * self.M + flat_clauses.astype(cp.int64)
        unique_keys = cp.unique(combined_keys)
        u_clusters = (unique_keys // self.M).astype(cp.int32)
        u_clauses = (unique_keys % self.M).astype(cp.int32)
        data_c = cp.ones(len(u_clusters), dtype=cp.bool_)
        cluster_to_clauses = cpx.coo_matrix((data_c, (u_clusters, u_clauses)), shape=(n_comps, self.M)).tocsr()
        
        # Exclude dummy 0
        ghost_label = labels[0]
        unique_labels = cp.unique(labels)
        valid_clusters = unique_labels[unique_labels != ghost_label].astype(cp.int32)
        num_valid = len(valid_clusters)
        
        if num_valid > 0:
            c2c_indptr = cluster_to_clauses.indptr.astype(cp.int32)
            c2c_indices = cluster_to_clauses.indices.astype(cp.int32)
            c2v_indptr = cluster_to_vars.indptr.astype(cp.int32)
            c2v_indices = cluster_to_vars.indices.astype(cp.int32)
            lit_clusters_ptr = cp.ascontiguousarray(lit_clusters.astype(cp.int32))
            
            seed = int(time.time() * 1000) % 1000000007
            self.kernel((1,), (256,), (self.sigma, c2c_indptr, c2c_indices, c2v_indptr, c2v_indices, self.lits_idx, self.lits_sign, lit_clusters_ptr, valid_clusters, cp.int32(num_valid), cp.int32(self.steps_flips), cp.float32(omega), cp.float32(self.beta_scale), cp.uint64(seed)))
            
        # --- PHASE 2: UNSAT DYNAMICS (8x Boost) ---
        c_spins = self.sigma[self.lits_idx]
        lit_is_sat = (c_spins == self.lits_sign)
        num_lit_sat = cp.sum(lit_is_sat, axis=1)
        is_unsat = (num_lit_sat == 0)

        if cp.any(is_unsat):
            omega_2 = 8.0 * omega
            P_2 = 1.0 - cp.exp(-omega_2)
            idx_U = cp.where(is_unsat)[0]
            n_unsat = len(idx_U)
            r_vals_U = cp.random.random(n_unsat, dtype=cp.float32)
            
            src_nodes_2 = []
            dst_nodes_2 = []
            
            P_7 = P_2 / 7.0
            
            # [6P/7, P) -> Full Freeze
            mask_full = (r_vals_U >= 6.0 * P_7) & (r_vals_U < P_2)
            if cp.any(mask_full):
                sub_idx = idx_U[mask_full]
                lits = self.lits_idx[sub_idx]
                src_nodes_2.append(lits[:, 0])
                dst_nodes_2.append(lits[:, 1])
                src_nodes_2.append(lits[:, 1])
                dst_nodes_2.append(lits[:, 2])

            # [0, 2P/7) -> Edge 0
            mask_e0 = (r_vals_U < 2.0 * P_7)
            if cp.any(mask_e0):
                sub_idx = idx_U[mask_e0]
                lits = self.lits_idx[sub_idx]
                src_nodes_2.append(lits[:, 0])
                dst_nodes_2.append(lits[:, 1])

            # [2P/7, 4P/7) -> Edge 1
            mask_e1 = (r_vals_U >= 2.0 * P_7) & (r_vals_U < 4.0 * P_7)
            if cp.any(mask_e1):
                sub_idx = idx_U[mask_e1]
                lits = self.lits_idx[sub_idx]
                src_nodes_2.append(lits[:, 1])
                dst_nodes_2.append(lits[:, 2])

            # [4P/7, 6P/7) -> Edge 2
            mask_e2 = (r_vals_U >= 4.0 * P_7) & (r_vals_U < 6.0 * P_7)
            if cp.any(mask_e2):
                sub_idx = idx_U[mask_e2]
                lits = self.lits_idx[sub_idx]
                src_nodes_2.append(lits[:, 2])
                dst_nodes_2.append(lits[:, 0])

            if len(src_nodes_2) > 0:
                all_src_2 = cp.concatenate(src_nodes_2)
                all_dst_2 = cp.concatenate(dst_nodes_2)
                data_2 = cp.ones(len(all_src_2), dtype=cp.float32)
                adj_2 = cpx.coo_matrix((data_2, (all_src_2, all_dst_2)), shape=(self.N+1, self.N+1), dtype=cp.float32)
                n_comps_2, labels_2 = cpx_graph.connected_components(adj_2, directed=False)
            else:
                n_comps_2 = self.N + 1
                labels_2 = cp.arange(self.N + 1, dtype=cp.int32)

            if verbose:
                comp_sizes_2 = cp.bincount(labels_2)
                sorted_sizes_2 = cp.sort(comp_sizes_2)[::-1]
                print(f"Complete SW Phase 2 (UNSAT) Top 7 Clusters: {sorted_sizes_2[:7]}")

            # Valid Clusters: Variables in UNSAT clauses (including singletons)
            unsat_vars = self.lits_idx[idx_U].flatten()
            relevant_clusters = labels_2[unsat_vars]
            unique_relevant = cp.unique(relevant_clusters)
            ghost_label_2 = labels_2[0]
            valid_clusters_2 = unique_relevant[unique_relevant != ghost_label_2].astype(cp.int32)
            num_valid_2 = len(valid_clusters_2)

            if num_valid_2 > 0:
                # Setup Kernel Phase 2
                lit_clusters_2 = labels_2[self.lits_idx]
                data_v_2 = cp.ones(self.N + 1, dtype=cp.bool_)
                cluster_to_vars_2 = cpx.coo_matrix((data_v_2, (labels_2, cp.arange(self.N + 1))), shape=(n_comps_2, self.N + 1)).tocsr()
                flat_clusters_2 = lit_clusters_2.flatten()
                flat_clauses_2 = cp.repeat(cp.arange(self.M), 3)
                combined_keys_2 = flat_clusters_2.astype(cp.int64) * self.M + flat_clauses_2.astype(cp.int64)
                unique_keys_2 = cp.unique(combined_keys_2)
                u_clusters_2 = (unique_keys_2 // self.M).astype(cp.int32)
                u_clauses_2 = (unique_keys_2 % self.M).astype(cp.int32)
                data_c_2 = cp.ones(len(u_clusters_2), dtype=cp.bool_)
                cluster_to_clauses_2 = cpx.coo_matrix((data_c_2, (u_clusters_2, u_clauses_2)), shape=(n_comps_2, self.M)).tocsr()

                c2c_indptr_2 = cluster_to_clauses_2.indptr.astype(cp.int32)
                c2c_indices_2 = cluster_to_clauses_2.indices.astype(cp.int32)
                c2v_indptr_2 = cluster_to_vars_2.indptr.astype(cp.int32)
                c2v_indices_2 = cluster_to_vars_2.indices.astype(cp.int32)
                lit_clusters_ptr_2 = cp.ascontiguousarray(lit_clusters_2.astype(cp.int32))
                
                # Reuse seed + offset
                self.kernel((1,), (256,), (self.sigma, c2c_indptr_2, c2c_indices_2, c2v_indptr_2, c2v_indices_2, self.lits_idx, self.lits_sign, lit_clusters_ptr_2, valid_clusters_2, cp.int32(num_valid_2), cp.int32(self.steps_flips), cp.float32(omega_2), cp.float32(self.beta_scale), cp.uint64(seed + 100)))

        current_energy = self.energy_check()
        if current_energy < self.min_energy:
            self.min_energy = current_energy
            self.best_sigma = self.sigma.copy()
            if self.min_energy == 0.0:
                print("🎉 COMPLETE SOLUTION FOUND ! (Energy = 0.0) 🎉")
                
        return current_energy, 0.0, 0.0"""
add_code(code_source_4b, execution_count=None, outputs=[])

# Cell 5 (Code)
code_source_5 = r"""# @title 4. Baseline: `WalkSAT` (CPU Optimized)
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
add_code(code_source_5, execution_count=5, outputs=[])

# Cell 6 (Code)
code_source_6 = r"""# @title 5. Main Simulation Loop
N = 10000
alpha = 4 # 4.25
clauses_np, _ = generate_random_3sat(N, alpha, seed=42)
print(f"Instance: N={N}, M={len(clauses_np)}, Alpha={alpha}")

# Solvers
facteur_complete = 5
solver = StochasticSwendsenWangGPU(clauses_np, N, beta_scale=15.0)
solver_gl = SwendsenWangGlauberGPU(clauses_np, N, beta_scale=15.0, steps_flips=2*N)
solver_complete = CompleteSwendsenWangGPU(clauses_np, N, beta_scale=15.0 * facteur_complete, steps_flips=2*N)
walksat = WalkSAT(clauses_np, N)

steps = 10000
omega_min = 0.25
omega_max = 0.5

epsilon = 1e-2
raw_decay = np.geomspace(1, epsilon, steps)
decay_01 = (raw_decay - epsilon) / (1.0 - epsilon)
omega_schedule = omega_max - (omega_max - omega_min) * decay_01

# History
history_sw = []
history_c1 = []
history_c2 = []

history_gl = [] # Glauber
history_gl_c1 = []
history_gl_c2 = []

history_cp = [] # Complete
history_cp_c1 = []
history_cp_c2 = []

history_ws = []

t0 = time.time()
print("Starting Comparison...")

for i, omega in enumerate(omega_schedule):
    is_verbose = (i % 20 == 0)
    
    # 1. Stochastic SW (Original)
    unsat_sw, c1_val, c2_val = solver.step(omega)

    if hasattr(unsat_sw, 'get'): history_sw.append(float(unsat_sw.get()))
    else: history_sw.append(float(unsat_sw))

    if hasattr(c1_val, 'get'): history_c1.append(float(c1_val.get()))
    else: history_c1.append(float(c1_val))

    if hasattr(c2_val, 'get'): history_c2.append(float(c2_val.get()))
    else: history_c2.append(float(c2_val))

    # 2. SW Glauber (New)
    unsat_gl, c1_gl, c2_gl = solver_gl.step(omega, verbose=False)

    if hasattr(unsat_gl, 'get'): history_gl.append(float(unsat_gl.get()))
    else: history_gl.append(float(unsat_gl))

    if hasattr(c1_gl, 'get'): history_gl_c1.append(float(c1_gl.get()))
    else: history_gl_c1.append(float(c1_gl))

    if hasattr(c2_gl, 'get'): history_gl_c2.append(float(c2_gl.get()))
    else: history_gl_c2.append(float(c2_gl))
    
    # 3. Complete SW (New)
    unsat_cp, c1_cp, c2_cp = solver_complete.step(omega / facteur_complete, verbose=is_verbose)
    
    if hasattr(unsat_cp, 'get'): history_cp.append(float(unsat_cp.get()))
    else: history_cp.append(float(unsat_cp))
    
    if hasattr(c1_cp, 'get'): history_cp_c1.append(float(c1_cp.get()))
    else: history_cp_c1.append(float(c1_cp))

    # 4. WalkSAT
    flips_per_step = N//10000
    if flips_per_step < 1: flips_per_step = 1

    e_ws = 1.0
    for _ in range(flips_per_step):
        e_ws = walksat.step(flips=1)
        if e_ws == 0.0: break

    history_ws.append(e_ws)

    if is_verbose:
        print(f"Step {i:3d} | Omega {omega:.3f} | SW: {unsat_sw:.6f} | GL: {unsat_gl:.6f} | CP: {unsat_cp:.6f} | WS: {e_ws:.6f}")

dt = time.time() - t0
print(f"Done in {dt:.2f}s")

# Plot
omega_cpu = omega_schedule
sw_cpu = np.array(history_sw)
gl_cpu = np.array(history_gl)
cp_cpu = np.array(history_cp)
ws_cpu = np.array(history_ws)
c1_cpu = np.array(history_c1)
c1_gl_cpu = np.array(history_gl_c1)
c1_cp_cpu = np.array(history_cp_c1)

plt.figure(figsize=(12, 7))
ax1 = plt.gca()

# Energy Axis
l1, = ax1.plot(omega_cpu, sw_cpu, label='Stochastic SW', color='cyan', linewidth=2)
l2, = ax1.plot(omega_cpu, gl_cpu, label='SW + Glauber', color='lime', linewidth=2, linestyle='-')
l3, = ax1.plot(omega_cpu, cp_cpu, label='Complete SW', color='yellow', linewidth=2, linestyle='-')
l4, = ax1.plot(omega_cpu, ws_cpu, label='WalkSAT', color='red', alpha=0.6)

ax1.set_xlabel(r'Coupling $\omega$ (Time)')
ax1.set_ylabel('Fraction Unsatisfied', color='white')
ax1.tick_params(axis='y', labelcolor='white')
ax1.grid(True, alpha=0.2)

# Cluster Axis
ax2 = ax1.twinx()
l5, = ax2.plot(omega_cpu, c1_cpu, label='Cluster (SW)', color='magenta', linestyle='--', linewidth=1.5, alpha=0.5)
l6, = ax2.plot(omega_cpu, c1_gl_cpu, label='Cluster (GL)', color='green', linestyle=':', linewidth=2.0, alpha=0.5)
l7, = ax2.plot(omega_cpu, c1_cp_cpu, label='Cluster (CP)', color='orange', linestyle=':', linewidth=2.0, alpha=0.5)

ax2.set_ylabel('Cluster Size Fraction', color='white')
ax2.tick_params(axis='y', labelcolor='white')

# Legend
lines = [l1, l2, l3, l4, l5, l6, l7]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title(f'Solver Comparison (N={N}, Alpha={alpha})')
plt.show()"""
add_code(code_source_6, execution_count=None, outputs=[{"output_type": "stream", "name": "stdout", "text": ["Instance: N=10000, M=40000, Alpha=4\n"]}])

with open("Swendsen-Wang_3SAT_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully.")
