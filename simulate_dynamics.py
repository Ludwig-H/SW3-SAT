import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import random

# Check for CuPy
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx
    import cupyx.scipy.sparse.csgraph as cpx_graph
    print(f"GPU Detected: {cp.cuda.runtime.getDeviceCount()} device(s)")
except ImportError:
    print("Error: CuPy not installed or no GPU detected. This script requires a GPU.")
    sys.exit(1)

plt.style.use('dark_background')

# --- KERNELS ---

metropolis_kernel_code = r'''
#include <curand_kernel.h>

extern "C" __global__ void run_metropolis_dynamics(
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
    __shared__ int delta_E_shared;
    __shared__ int decision_shared;
    __shared__ int target_cluster_shared;

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

        // --- 2. Compute Delta E ---
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

dynamics_unsat_kernel_code = r'''
#include <curand_kernel.h>

extern "C" __global__ void run_dynamics_unsat(
    signed char* sigma,           // N
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
    unsigned long long seed,
    int require_unsat             // 0 = Normal, 1 = Only clusters touching UNSAT clauses
) {
    __shared__ int delta_E_shared;
    __shared__ int decision_shared;
    __shared__ int target_cluster_shared;
    __shared__ int is_cluster_active; 
    __shared__ int step_valid;

    curandState state;
    if (threadIdx.x == 0) curand_init(seed, 0, 0, &state);

    int active_steps = 0;
    int safety_counter = 0;
    int max_safety = steps * 200; // Prevent infinite loop if no valid clusters exist

    // If requiring UNSAT, use WHILE loop behavior. Otherwise FOR loop.
    while (active_steps < steps) {
        
        // Safety Break
        if (require_unsat && safety_counter++ > max_safety) break;

        __syncthreads();

        // --- 1. Pick Target Cluster ---
        if (threadIdx.x == 0) {
            delta_E_shared = 0;
            decision_shared = 0;
            is_cluster_active = (require_unsat) ? 0 : 1; 
            step_valid = (require_unsat) ? 0 : 1;
            
            unsigned int r = curand(&state);
            int r_idx = r % num_valid;
            target_cluster_shared = valid_clusters[r_idx];
        }
        __syncthreads();

        int c_id = target_cluster_shared;
        int start_c = c2c_indptr[c_id];
        int end_c = c2c_indptr[c_id+1];

        // --- 2. DYNAMIC CHECK ---
        if (require_unsat && start_c < end_c) {
            for (int i = start_c + threadIdx.x; i < end_c; i += blockDim.x) {
                if (is_cluster_active) break; 

                int clause_idx = c2c_indices[i];
                int idx0 = clause_idx * 3 + 0;
                int idx1 = clause_idx * 3 + 1;
                int idx2 = clause_idx * 3 + 2;

                bool sat = (sigma[lits_idx[idx0]] == lits_sign[idx0]) ||
                           (sigma[lits_idx[idx1]] == lits_sign[idx1]) ||
                           (sigma[lits_idx[idx2]] == lits_sign[idx2]);
                
                if (!sat) {
                    is_cluster_active = 1; 
                    step_valid = 1;
                }
            }
        }
        __syncthreads();

        if (is_cluster_active == 0) continue; 

        // If we passed the check, this is a valid step
        if (threadIdx.x == 0) {
            active_steps++; 
        }

        // --- 3. Compute Delta E ---
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
                    atomicAdd(&delta_E_shared, (int)sat_curr - (int)sat_new);
                }
            }
        }
        __syncthreads();

        // --- 4. Decision ---
        if (threadIdx.x == 0) {
            int dE = delta_E_shared;
            if (dE <= 0) {
                decision_shared = 1;
            } else {
                float p = expf(-(float)dE * omega * beta_scale);
                if (curand_uniform(&state) < p) {
                    decision_shared = 1;
                }
            }
        }
        __syncthreads();

        // --- 5. Update ---
        if (decision_shared) {
            int start_v = c2v_indptr[c_id];
            int end_v = c2v_indptr[c_id+1];
            for (int i = start_v + threadIdx.x; i < end_v; i += blockDim.x) {
                sigma[c2v_indices[i]] *= -1;
            }
        }
    }
}
'''

# --- GENERATOR ---

def generate_random_3sat(N, alpha, seed=None):
    if seed is not None: np.random.seed(seed)
    M = int(N * alpha)
    vars = np.random.randint(1, N + 1, size=(M, 3))
    signs = np.random.choice([-1, 1], size=(M, 3))
    return vars * signs, N

# --- SOLVERS ---

class SwendsenWangErdosRenyiGPU:
    def __init__(self, clauses_np, N, beta_scale=15.0, steps_flips=None, a=0.9, dynamics="Metropolis"):
        self.N = N
        self.M = len(clauses_np)
        self.clauses = cp.array(clauses_np)
        self.beta_scale = beta_scale
        self.a = a 
        self.dynamics = dynamics
        
        if steps_flips is None:
            self.steps_flips = 2 * N
        else:
            self.steps_flips = steps_flips

        self.lits_idx = cp.ascontiguousarray(cp.abs(self.clauses).astype(cp.int32) - 1)
        self.lits_sign = cp.ascontiguousarray(cp.sign(self.clauses).astype(cp.int8))

        s = self.lits_sign
        j01 = cp.where(s[:, 0] == s[:, 1], -1, 1)
        j12 = cp.where(s[:, 1] == s[:, 2], -1, 1)
        j20 = cp.where(s[:, 2] == s[:, 0], -1, 1)
        self.J_tri = cp.stack([j01, j12, j20], axis=1).astype(cp.int8)

        self.sigma = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=N)
        
        self.best_sigma = self.sigma.copy()
        self.min_energy = 1.0

        self.kernel = cp.RawKernel(metropolis_kernel_code, 'run_metropolis_dynamics', options=('-std=c++17',))

    def energy_check(self):
        spins = self.sigma[self.lits_idx]
        is_lit_sat = (spins == self.lits_sign)
        is_clause_sat = cp.any(is_lit_sat, axis=1)
        return 1.0 - cp.mean(is_clause_sat)


    def _run_dynamics(self, labels, n_comps, omega):
        lit_clusters = labels[self.lits_idx]
        valid_clusters = cp.unique(labels).astype(cp.int32)
        valid_clusters = valid_clusters[valid_clusters >= 0]
        num_valid = len(valid_clusters)
        if num_valid == 0: return

        valid_mask_v = (labels >= 0)
        active_vars = cp.where(valid_mask_v)[0]
        active_labels = labels[valid_mask_v]
        if len(active_vars) == 0: return
        data_v = cp.ones(len(active_vars), dtype=cp.bool_)
        cluster_to_vars = cpx.coo_matrix((data_v, (active_labels, active_vars)), shape=(n_comps, self.N)).tocsr()

        flat_clusters = lit_clusters.flatten()
        flat_clauses = cp.repeat(cp.arange(self.M), 3)
        mask_c = (flat_clusters >= 0)
        flat_clusters = flat_clusters[mask_c]
        flat_clauses = flat_clauses[mask_c]
        if len(flat_clusters) == 0: return
        combined_keys = flat_clusters.astype(cp.int64) * self.M + flat_clauses.astype(cp.int64)
        unique_keys = cp.unique(combined_keys)
        u_clusters = (unique_keys // self.M).astype(cp.int32)
        u_clauses = (unique_keys % self.M).astype(cp.int32)
        data_c = cp.ones(len(u_clusters), dtype=cp.bool_)
        cluster_to_clauses = cpx.coo_matrix((data_c, (u_clusters, u_clauses)), shape=(n_comps, self.M)).tocsr()

        c2c_indptr = cluster_to_clauses.indptr.astype(cp.int32)
        c2c_indices = cluster_to_clauses.indices.astype(cp.int32)
        c2v_indptr = cluster_to_vars.indptr.astype(cp.int32)
        c2v_indices = cluster_to_vars.indices.astype(cp.int32)
        lit_clusters_ptr = cp.ascontiguousarray(lit_clusters.astype(cp.int32))
        
        seed = int(time.time() * 1000) % 1000000007

        self.kernel(
            (1,), (256,),
            (
                self.sigma, c2c_indptr, c2c_indices, c2v_indptr, c2v_indices,
                self.lits_idx, self.lits_sign, lit_clusters_ptr, valid_clusters,
                cp.int32(num_valid), cp.int32(self.steps_flips),
                cp.float32(omega), cp.float32(self.beta_scale), cp.uint64(seed)
            )
        )


    def step(self, omega, verbose=False):
        c_spins = self.sigma[self.lits_idx]
        s0, s1, s2 = c_spins[:, 0], c_spins[:, 1], c_spins[:, 2]
        sat0 = (s0 * s1 * self.J_tri[:, 0] == 1)
        sat1 = (s1 * s2 * self.J_tri[:, 1] == 1)
        sat2 = (s2 * s0 * self.J_tri[:, 2] == 1)
        sat_mask = cp.stack([sat0, sat1, sat2], axis=1)
        num_sat_tri = cp.sum(sat_mask, axis=1)
        is_low_energy = (num_sat_tri == 2)

        P = 1.0 - cp.exp(-omega)
        rand_vals = cp.random.random(self.M, dtype=cp.float32)
        src_nodes, dst_nodes = [], []

        mask_T = is_low_energy & (rand_vals < P)
        if cp.any(mask_T):
            idx_T = cp.where(mask_T)[0]
            r_vals_T = rand_vals[idx_T]
            sub_sat = sat_mask[idx_T]
            idx_1st = cp.argmax(sub_sat, axis=1)
            idx_sum = cp.sum(sub_sat * cp.array([0, 1, 2], dtype=cp.int8), axis=1)
            idx_2nd = idx_sum - idx_1st
            pick_first = (r_vals_T < (P / 2.0))
            chosen = cp.where(pick_first, idx_1st, idx_2nd)
            lits = self.lits_idx[idx_T]
            l0, l1, l2 = lits[:,0], lits[:,1], lits[:,2]
            s_e = cp.where(chosen==0, l0, cp.where(chosen==1, l1, l2))
            d_e = cp.where(chosen==0, l1, cp.where(chosen==1, l2, l0))
            src_nodes.append(s_e)
            dst_nodes.append(d_e)

        if len(src_nodes) > 0:
            all_src = cp.concatenate(src_nodes)
            all_dst = cp.concatenate(dst_nodes)
            data = cp.ones(len(all_src), dtype=cp.float32)
            adj = cpx.coo_matrix((data, (all_src, all_dst)), shape=(self.N, self.N))
            n_comps, labels = cpx_graph.connected_components(adj, directed=False)
        else:
            n_comps = self.N
            labels = cp.arange(self.N, dtype=cp.int32)

        if verbose:
            comp_sizes = cp.bincount(labels)
            sorted_sizes = cp.sort(comp_sizes)[::-1]
            top20 = sorted_sizes[:20].get() if hasattr(sorted_sizes, 'get') else sorted_sizes[:20]
            print(f"Phase 1: {n_comps} clusters. Top 20 sizes: {top20}")

        self._run_dynamics(labels, n_comps, omega)
        
        c_spins = self.sigma[self.lits_idx]
        lit_is_sat = (c_spins == self.lits_sign)
        is_unsat = (cp.sum(lit_is_sat, axis=1) == 0)
        
        if cp.any(is_unsat):
            omega_2 = 8.0 * omega
            idx_U = cp.where(is_unsat)[0]
            n_unsat = len(idx_U)
            r_vals_U = cp.random.random(n_unsat, dtype=cp.float32)
            src_2, dst_2 = [], []
            T1, T2, T3 = 2.0 / 7.0, 4.0 / 7.0, 6.0 / 7.0
            
            mask_full = (r_vals_U >= T3)
            if cp.any(mask_full):
                l = self.lits_idx[idx_U[mask_full]]
                src_2.append(l[:,0]); dst_2.append(l[:,1])
                src_2.append(l[:,1]); dst_2.append(l[:,2])
            
            mask_e0 = (r_vals_U < T1)
            if cp.any(mask_e0):
                l = self.lits_idx[idx_U[mask_e0]]
                src_2.append(l[:,0]); dst_2.append(l[:,1])
                
            mask_e1 = (r_vals_U >= T1) & (r_vals_U < T2)
            if cp.any(mask_e1):
                l = self.lits_idx[idx_U[mask_e1]]
                src_2.append(l[:,1]); dst_2.append(l[:,2])
                
            mask_e2 = (r_vals_U >= T2) & (r_vals_U < T3)
            if cp.any(mask_e2):
                l = self.lits_idx[idx_U[mask_e2]]
                src_2.append(l[:,2]); dst_2.append(l[:,0])
            if len(src_2) > 0:
                all_src_2 = cp.concatenate(src_2)
                all_dst_2 = cp.concatenate(dst_2)
                data_2 = cp.ones(len(all_src_2), dtype=cp.float32)
                adj_2 = cpx.coo_matrix((data_2, (all_src_2, all_dst_2)), shape=(self.N, self.N))
                n_comps_2, labels_2 = cpx_graph.connected_components(adj_2, directed=False)
            else:
                n_comps_2 = self.N
                labels_2 = cp.arange(self.N, dtype=cp.int32)
                
            unsat_vars = self.lits_idx[idx_U].flatten()
            active_clusters = labels_2[unsat_vars]
            unique_active = cp.unique(active_clusters)
            m = len(unique_active)
            final_labels = labels_2
            final_n_comps = n_comps_2
            
            if m > 1 and self.a > 0:
                cluster_map = cp.full(n_comps_2, -1, dtype=cp.int32)
                cluster_map[unique_active] = cp.arange(m, dtype=cp.int32)
                num_edges = int(self.a * (m - 1) / 2)
                if num_edges > 0:
                    s_er = cp.random.randint(0, m, size=num_edges, dtype=cp.int32)
                    d_er = cp.random.randint(0, m, size=num_edges, dtype=cp.int32)
                    data_er = cp.ones(num_edges, dtype=cp.float32)
                    adj_er = cpx.coo_matrix((data_er, (s_er, d_er)), shape=(m, m))
                    n_super, super_labels = cpx_graph.connected_components(adj_er, directed=False)
                    mapped_ids = cluster_map[labels_2]
                    is_active = (mapped_ids != -1)
                    new_labels = cp.zeros(self.N, dtype=cp.int32)
                    new_labels[:] = -1
                    new_labels[is_active] = super_labels[mapped_ids[is_active]]
                    final_labels = new_labels
                    final_n_comps = n_super
            
            if verbose:
                active_mask = (final_labels != -1)
                if cp.any(active_mask):
                    comp_sizes_2 = cp.bincount(final_labels[active_mask])
                    sorted_sizes_2 = cp.sort(comp_sizes_2)[::-1]
                    top20_2 = sorted_sizes_2[:20].get() if hasattr(sorted_sizes_2, 'get') else sorted_sizes_2[:20]
                    print(f"Phase 2: {n_comps_2} clusters -> {final_n_comps} super-clusters (Active). Top 20 sizes: {top20_2}")
                else:
                    print("Phase 2: No active clusters.")

            self._run_dynamics(final_labels, final_n_comps, omega_2)

        e = self.energy_check()
        if e < self.min_energy:
            self.min_energy = e
            self.best_sigma = self.sigma.copy()
            if e == 0.0:
                print(f"🎉 SOLUTION FOUND ! (Energy = 0.0) 🎉")
        
        return e, 0.0, 0.0

class ConstrainedSwendsenWangErdosRenyiGPU:
    def __init__(self, clauses_np, N, beta_scale=15.0, steps_flips=None, a=0.9, dynamics="Metropolis"):
        self.N = N
        self.M = len(clauses_np)
        self.clauses = cp.array(clauses_np)
        self.beta_scale = beta_scale
        self.a = a
        self.dynamics = dynamics
        
        if steps_flips is None:
            self.steps_flips = 2 * N
        else:
            self.steps_flips = steps_flips

        self.lits_idx = cp.ascontiguousarray(cp.abs(self.clauses).astype(cp.int32))
        self.lits_sign = cp.ascontiguousarray(cp.sign(self.clauses).astype(cp.int8))

        s = self.lits_sign
        j01 = cp.where(s[:, 0] == s[:, 1], -1, 1)
        j12 = cp.where(s[:, 1] == s[:, 2], -1, 1)
        j20 = cp.where(s[:, 2] == s[:, 0], -1, 1)
        self.J_tri = cp.stack([j01, j12, j20], axis=1).astype(cp.int8)

        self.sigma = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=N+1)
        self.sigma[0] = 1 
        
        self.best_sigma = self.sigma.copy()
        self.min_energy = 1.0

        self.kernel = cp.RawKernel(metropolis_kernel_code, 'run_metropolis_dynamics', options=('-std=c++17',))

    def energy_check(self):
        spins = self.sigma[self.lits_idx]
        is_lit_sat = (spins == self.lits_sign)
        is_clause_sat = cp.any(is_lit_sat, axis=1)
        return 1.0 - cp.mean(is_clause_sat)

    def _run_dynamics(self, labels, n_comps, omega):
        lit_clusters = labels[self.lits_idx]
        ghost_label = labels[0]
        valid_clusters = cp.unique(labels).astype(cp.int32)
        valid_clusters = valid_clusters[(valid_clusters != ghost_label) & (valid_clusters >= 0)]
        num_valid = len(valid_clusters)
        
        if num_valid == 0: return

        valid_mask_v = (labels >= 0)
        active_vars = cp.where(valid_mask_v)[0]
        active_labels = labels[valid_mask_v]
        
        if len(active_vars) == 0: return

        data_v = cp.ones(len(active_vars), dtype=cp.bool_)
        cluster_to_vars = cpx.coo_matrix(
            (data_v, (active_labels, active_vars)),
            shape=(n_comps, self.N + 1)
        ).tocsr()

        flat_clusters = lit_clusters.flatten()
        flat_clauses = cp.repeat(cp.arange(self.M), 3)
        mask_c = (flat_clusters >= 0)
        flat_clusters = flat_clusters[mask_c]
        flat_clauses = flat_clauses[mask_c]
        if len(flat_clusters) == 0: return

        combined_keys = flat_clusters.astype(cp.int64) * self.M + flat_clauses.astype(cp.int64)
        unique_keys = cp.unique(combined_keys)
        u_clusters = (unique_keys // self.M).astype(cp.int32)
        u_clauses = (unique_keys % self.M).astype(cp.int32)
        data_c = cp.ones(len(u_clusters), dtype=cp.bool_)
        cluster_to_clauses = cpx.coo_matrix((data_c, (u_clusters, u_clauses)), shape=(n_comps, self.M)).tocsr()

        c2c_indptr = cluster_to_clauses.indptr.astype(cp.int32)
        c2c_indices = cluster_to_clauses.indices.astype(cp.int32)
        c2v_indptr = cluster_to_vars.indptr.astype(cp.int32)
        c2v_indices = cluster_to_vars.indices.astype(cp.int32)
        lit_clusters_ptr = cp.ascontiguousarray(lit_clusters.astype(cp.int32))
        
        seed = int(time.time() * 1000) % 1000000007

        self.kernel(
            (1,), (256,),
            (
                self.sigma, c2c_indptr, c2c_indices, c2v_indptr, c2v_indices,
                self.lits_idx, self.lits_sign, lit_clusters_ptr, valid_clusters,
                cp.int32(num_valid), cp.int32(self.steps_flips),
                cp.float32(omega), cp.float32(self.beta_scale), cp.uint64(seed)
            )
        )

    def step(self, omega, verbose=False):
        c_spins = self.sigma[self.lits_idx]
        lit_is_sat = (c_spins == self.lits_sign)
        is_clause_sat = cp.any(lit_is_sat, axis=1)
        
        is_unsat_clause = ~is_clause_sat
        marked_vars = cp.zeros(self.N + 1, dtype=bool)
        if cp.any(is_unsat_clause):
            unsat_vars = self.lits_idx[is_unsat_clause].flatten()
            marked_vars[unsat_vars] = True
            
        lit_marked = marked_vars[self.lits_idx] 
        num_marked = cp.sum(lit_marked, axis=1)

        s0, s1, s2 = c_spins[:, 0], c_spins[:, 1], c_spins[:, 2]
        sat0 = (s0 * s1 * self.J_tri[:, 0] == 1)
        sat1 = (s1 * s2 * self.J_tri[:, 1] == 1)
        sat2 = (s2 * s0 * self.J_tri[:, 2] == 1)
        sat_mask = cp.stack([sat0, sat1, sat2], axis=1)
        num_sat_tri = cp.sum(sat_mask, axis=1)
        
        num_lit_sat = cp.sum(lit_is_sat, axis=1)
        is_fully_sat = (num_lit_sat == 3)
        is_low_energy = (num_sat_tri == 2)

        P = 1.0 - cp.exp(-omega)
        rand_vals = cp.random.random(self.M, dtype=cp.float32)
        src_nodes, dst_nodes = [], []

        mask_B = is_low_energy & (rand_vals < P)
        if cp.any(mask_B):
            idx_B = cp.where(mask_B)[0]
            nm_B = num_marked[idx_B]
            
            mask_B1 = (nm_B == 0)
            if cp.any(mask_B1):
                idx_B1 = idx_B[mask_B1]
                sub_sat = sat_mask[idx_B1]
                idx_1st = cp.argmax(sub_sat, axis=1)
                idx_sum = cp.sum(sub_sat * cp.array([0, 1, 2], dtype=cp.int8), axis=1)
                idx_2nd = idx_sum - idx_1st
                pick_first = (rand_vals[idx_B1] < (P / 2.0))
                chosen = cp.where(pick_first, idx_1st, idx_2nd)
                lits = self.lits_idx[idx_B1]
                l0, l1, l2 = lits[:,0], lits[:,1], lits[:,2]
                s_e = cp.where(chosen==0, l0, cp.where(chosen==1, l1, l2))
                d_e = cp.where(chosen==0, l1, cp.where(chosen==1, l2, l0))
                src_nodes.append(s_e); dst_nodes.append(d_e)
                
            mask_B2 = (nm_B == 1)
            if cp.any(mask_B2):
                idx_B2 = idx_B[mask_B2]
                marked_col = cp.argmax(lit_marked[idx_B2], axis=1)
                opp_edge = (marked_col + 1) % 3
                is_opp_sat = sat_mask[idx_B2, opp_edge]
                if cp.any(is_opp_sat):
                    sub_idx = idx_B2[is_opp_sat]
                    target_edge = opp_edge[is_opp_sat]
                    lits = self.lits_idx[sub_idx]
                    l0, l1, l2 = lits[:,0], lits[:,1], lits[:,2]
                    s_e = cp.where(target_edge==0, l0, cp.where(target_edge==1, l1, l2))
                    d_e = cp.where(target_edge==0, l1, cp.where(target_edge==1, l2, l0))
                    src_nodes.append(s_e); dst_nodes.append(d_e)

        if len(src_nodes) > 0:
            all_src = cp.concatenate(src_nodes)
            all_dst = cp.concatenate(dst_nodes)
            data = cp.ones(len(all_src), dtype=cp.float32)
            adj = cpx.coo_matrix((data, (all_src, all_dst)), shape=(self.N + 1, self.N + 1))
            n_comps, labels = cpx_graph.connected_components(adj, directed=False)
        else:
            n_comps = self.N + 1
            labels = cp.arange(self.N + 1, dtype=cp.int32)

        if verbose:
            comp_sizes = cp.bincount(labels)
            sorted_sizes = cp.sort(comp_sizes)[::-1]
            top20 = sorted_sizes[:20].get() if hasattr(sorted_sizes, 'get') else sorted_sizes[:20]
            print(f"Phase 1 (Constrained): {n_comps} clusters. Top 20 sizes: {top20}")

        self._run_dynamics(labels, n_comps, omega)
        
        c_spins = self.sigma[self.lits_idx]
        lit_is_sat = (c_spins == self.lits_sign)
        is_unsat = (cp.sum(lit_is_sat, axis=1) == 0)
        
        if cp.any(is_unsat):
            omega_2 = 8.0 * omega
            idx_U = cp.where(is_unsat)[0]
            n_unsat = len(idx_U)
            r_vals_U = cp.random.random(n_unsat, dtype=cp.float32)
            src_2, dst_2 = [], []
            T1, T2, T3 = 2.0 / 7.0, 4.0 / 7.0, 6.0 / 7.0
            
            mask_full = (r_vals_U >= T3)
            if cp.any(mask_full):
                l = self.lits_idx[idx_U[mask_full]]
                src_2.append(l[:,0]); dst_2.append(l[:,1])
                src_2.append(l[:,1]); dst_2.append(l[:,2])
            mask_e0 = (r_vals_U < T1)
            if cp.any(mask_e0):
                l = self.lits_idx[idx_U[mask_e0]]
                src_2.append(l[:,0]); dst_2.append(l[:,1])
            mask_e1 = (r_vals_U >= T1) & (r_vals_U < T2)
            if cp.any(mask_e1):
                l = self.lits_idx[idx_U[mask_e1]]
                src_2.append(l[:,1]); dst_2.append(l[:,2])
            mask_e2 = (r_vals_U >= T2) & (r_vals_U < T3)
            if cp.any(mask_e2):
                l = self.lits_idx[idx_U[mask_e2]]
                src_2.append(l[:,2]); dst_2.append(l[:,0])
            if len(src_2) > 0:
                all_src_2 = cp.concatenate(src_2)
                all_dst_2 = cp.concatenate(dst_2)
                data_2 = cp.ones(len(all_src_2), dtype=cp.float32)
                adj_2 = cpx.coo_matrix((data_2, (all_src_2, all_dst_2)), shape=(self.N + 1, self.N + 1))
                n_comps_2, labels_2 = cpx_graph.connected_components(adj_2, directed=False)
            else:
                n_comps_2 = self.N + 1
                labels_2 = cp.arange(self.N + 1, dtype=cp.int32)
                
            unsat_vars = self.lits_idx[idx_U].flatten()
            active_clusters = labels_2[unsat_vars]
            unique_active = cp.unique(active_clusters)
            ghost_l2 = labels_2[0]
            unique_active = unique_active[unique_active != ghost_l2]
            m = len(unique_active)
            final_labels = labels_2
            final_n_comps = n_comps_2
            if m > 1 and self.a > 0:
                cluster_map = cp.full(n_comps_2, -1, dtype=cp.int32)
                cluster_map[unique_active] = cp.arange(m, dtype=cp.int32)
                num_edges = int(self.a * (m - 1) / 2)
                if num_edges > 0:
                    s_er = cp.random.randint(0, m, size=num_edges, dtype=cp.int32)
                    d_er = cp.random.randint(0, m, size=num_edges, dtype=cp.int32)
                    data_er = cp.ones(num_edges, dtype=cp.float32)
                    adj_er = cpx.coo_matrix((data_er, (s_er, d_er)), shape=(m, m))
                    n_super, super_labels = cpx_graph.connected_components(adj_er, directed=False)
                    mapped_ids = cluster_map[labels_2]
                    is_active = (mapped_ids != -1)
                    new_labels = cp.full(self.N + 1, -1, dtype=cp.int32)
                    new_labels[is_active] = super_labels[mapped_ids[is_active]]
                    final_labels = new_labels
                    final_n_comps = n_super
            if verbose:
                active_mask = (final_labels != -1)
                if cp.any(active_mask):
                    comp_sizes_2 = cp.bincount(final_labels[active_mask])
                    sorted_sizes_2 = cp.sort(comp_sizes_2)[::-1]
                    top20_2 = sorted_sizes_2[:20].get() if hasattr(sorted_sizes_2, 'get') else sorted_sizes_2[:20]
                    print(f"Phase 2: {n_comps_2} clusters -> {final_n_comps} super-clusters (Active). Top 20 sizes: {top20_2}")

            self._run_dynamics(final_labels, final_n_comps, omega_2)

        e = self.energy_check()
        if e < self.min_energy:
            self.min_energy = e
            self.best_sigma = self.sigma.copy()
            if e == 0.0:
                print(f"🎉 SOLUTION FOUND ! (Energy = 0.0) 🎉")
        
        return e, 0.0, 0.0

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
        
        self.best_sigma = self.sigma.copy()
        self.min_energy = 1.0

        self.kernel = cp.RawKernel(metropolis_kernel_code, 'run_metropolis_dynamics', options=('-std=c++17',))

    def energy_check(self):
        spins = self.sigma[self.lits_idx]
        is_lit_sat = (spins == self.lits_sign)
        is_clause_sat = cp.any(is_lit_sat, axis=1)
        return 1.0 - cp.mean(is_clause_sat)

    def step(self, omega, verbose=False):
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

        mask_G = is_fully_sat & (rand_vals < P)
        if cp.any(mask_G):
            idx_G = cp.where(mask_G)[0]
            targets = self.lits_idx[idx_G].flatten()
            src_nodes.append(cp.zeros_like(targets))
            dst_nodes.append(targets)

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
            print(f"Phase 1 Top 7 Clusters: {sorted_sizes[:7]}")

        lit_clusters = labels[self.lits_idx]

        data_v = cp.ones(self.N + 1, dtype=cp.bool_)
        cluster_to_vars = cpx.coo_matrix(
            (data_v, (labels, cp.arange(self.N + 1))),
            shape=(n_comps, self.N + 1)
        ).tocsr()

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

        c_spins = self.sigma[self.lits_idx]
        lit_is_sat = (c_spins == self.lits_sign)
        num_lit_sat = cp.sum(lit_is_sat, axis=1)
        is_unsat = (num_lit_sat == 0)

        if cp.any(is_unsat):
            idx_U = cp.where(is_unsat)[0]
            n_unsat = len(idx_U)
            r_vals_U = cp.random.random(n_unsat, dtype=cp.float32)
            src_nodes_2, dst_nodes_2 = [], []
            P_7 = P / 7.0
            
            mask_full = (r_vals_U >= 6.0 * P_7) & (r_vals_U < P)
            if cp.any(mask_full):
                sub_idx = idx_U[mask_full]
                lits = self.lits_idx[sub_idx]
                src_nodes_2.append(lits[:, 0]); dst_nodes_2.append(lits[:, 1])
                src_nodes_2.append(lits[:, 1]); dst_nodes_2.append(lits[:, 2])
            
            mask_e0 = (r_vals_U < 2.0 * P_7)
            if cp.any(mask_e0):
                sub_idx = idx_U[mask_e0]
                lits = self.lits_idx[sub_idx]
                src_nodes_2.append(lits[:, 0]); dst_nodes_2.append(lits[:, 1])

            mask_e1 = (r_vals_U >= 2.0 * P_7) & (r_vals_U < 4.0 * P_7)
            if cp.any(mask_e1):
                sub_idx = idx_U[mask_e1]
                lits = self.lits_idx[sub_idx]
                src_nodes_2.append(lits[:, 1]); dst_nodes_2.append(lits[:, 2])

            mask_e2 = (r_vals_U >= 4.0 * P_7) & (r_vals_U < 6.0 * P_7)
            if cp.any(mask_e2):
                sub_idx = idx_U[mask_e2]
                lits = self.lits_idx[sub_idx]
                src_nodes_2.append(lits[:, 2]); dst_nodes_2.append(lits[:, 0])

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

        ghost_label_2 = labels_2[0]
        unique_labels_2 = cp.unique(labels_2)
        mask_valid = (comp_sizes_2[unique_labels_2] > 1) & (unique_labels_2 != ghost_label_2)
        valid_clusters_2 = unique_labels_2[mask_valid].astype(cp.int32)
        num_valid_2 = len(valid_clusters_2)

        if num_valid_2 > 0:
            c2c_indptr_2 = cluster_to_clauses_2.indptr.astype(cp.int32)
            c2c_indices_2 = cluster_to_clauses_2.indices.astype(cp.int32)
            c2v_indptr_2 = cluster_to_vars_2.indptr.astype(cp.int32)
            c2v_indices_2 = cluster_to_vars_2.indices.astype(cp.int32)
            lit_clusters_ptr_2 = cp.ascontiguousarray(lit_clusters_2.astype(cp.int32))
            
            self.kernel((1,), (256,), (self.sigma, c2c_indptr_2, c2c_indices_2, c2v_indptr_2, c2v_indices_2, self.lits_idx, self.lits_sign, lit_clusters_ptr_2, valid_clusters_2, cp.int32(num_valid_2), cp.int32(self.steps_flips), cp.float32(omega), cp.float32(self.beta_scale), cp.uint64(seed + 1)))

        current_energy = self.energy_check()
        
        if current_energy < self.min_energy:
            self.min_energy = current_energy
            self.best_sigma = self.sigma.copy()
            if self.min_energy == 0.0:
                print(f"🎉 SOLUTION FOUND ! (Energy = 0.0) 🎉")
        
        return current_energy, c1_frac, c2_frac

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
        self.kernel = cp.RawKernel(metropolis_kernel_code, 'run_metropolis_dynamics', options=('-std=c++17',))

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
            
            mask_full = (r_vals_U >= 6.0 * P_7) & (r_vals_U < P_2)
            if cp.any(mask_full):
                sub_idx = idx_U[mask_full]
                lits = self.lits_idx[sub_idx]
                src_nodes_2.append(lits[:, 0]); dst_nodes_2.append(lits[:, 1])
                src_nodes_2.append(lits[:, 1]); dst_nodes_2.append(lits[:, 2])

            mask_e0 = (r_vals_U < 2.0 * P_7)
            if cp.any(mask_e0):
                sub_idx = idx_U[mask_e0]
                lits = self.lits_idx[sub_idx]
                src_nodes_2.append(lits[:, 0]); dst_nodes_2.append(lits[:, 1])

            mask_e1 = (r_vals_U >= 2.0 * P_7) & (r_vals_U < 4.0 * P_7)
            if cp.any(mask_e1):
                sub_idx = idx_U[mask_e1]
                lits = self.lits_idx[sub_idx]
                src_nodes_2.append(lits[:, 1]); dst_nodes_2.append(lits[:, 2])

            mask_e2 = (r_vals_U >= 4.0 * P_7) & (r_vals_U < 6.0 * P_7)
            if cp.any(mask_e2):
                sub_idx = idx_U[mask_e2]
                lits = self.lits_idx[sub_idx]
                src_nodes_2.append(lits[:, 2]); dst_nodes_2.append(lits[:, 0])

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

            unsat_vars = self.lits_idx[idx_U].flatten()
            relevant_clusters = labels_2[unsat_vars]
            unique_relevant = cp.unique(relevant_clusters)
            ghost_label_2 = labels_2[0]
            valid_clusters_2 = unique_relevant[unique_relevant != ghost_label_2].astype(cp.int32)
            num_valid_2 = len(valid_clusters_2)

            if num_valid_2 > 0:
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
                
                self.kernel((1,), (256,), (self.sigma, c2c_indptr_2, c2c_indices_2, c2v_indptr_2, c2v_indices_2, self.lits_idx, self.lits_sign, lit_clusters_ptr_2, valid_clusters_2, cp.int32(num_valid_2), cp.int32(self.steps_flips), cp.float32(omega_2), cp.float32(self.beta_scale), cp.uint64(seed + 100)))

        current_energy = self.energy_check()
        if current_energy < self.min_energy:
            self.min_energy = current_energy
            self.best_sigma = self.sigma.copy()
            if self.min_energy == 0.0:
                print("🎉 COMPLETE SOLUTION FOUND ! (Energy = 0.0) 🎉")
                
        return current_energy, 0.0, 0.0

class DynamicsUNSAT_GPU:
    def __init__(self, clauses_np, N, beta_scale=15.0, steps_flips=None, a=0.9):
        self.N = N
        self.M = len(clauses_np)
        self.clauses = cp.array(clauses_np)
        self.beta_scale = beta_scale
        self.a = a
        if steps_flips is None:
            self.steps_flips = 2 * N
        else:
            self.steps_flips = steps_flips

        self.lits_idx = cp.ascontiguousarray(cp.abs(self.clauses).astype(cp.int32) - 1)
        self.lits_sign = cp.ascontiguousarray(cp.sign(self.clauses).astype(cp.int8))

        s = self.lits_sign
        j01 = cp.where(s[:, 0] == s[:, 1], -1, 1)
        j12 = cp.where(s[:, 1] == s[:, 2], -1, 1)
        j20 = cp.where(s[:, 2] == s[:, 0], -1, 1)
        self.J_tri = cp.stack([j01, j12, j20], axis=1).astype(cp.int8)

        self.sigma = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=N)
        self.best_sigma = self.sigma.copy()
        self.min_energy = 1.0
        self.kernel = cp.RawKernel(dynamics_unsat_kernel_code, 'run_dynamics_unsat', options=('-std=c++17',))

    def energy_check(self):
        spins = self.sigma[self.lits_idx]
        is_lit_sat = (spins == self.lits_sign)
        is_clause_sat = cp.any(is_lit_sat, axis=1)
        return 1.0 - cp.mean(is_clause_sat)

    def _run_dynamics(self, labels, n_comps, omega, require_unsat=0):
        lit_clusters = labels[self.lits_idx]
        valid_clusters = cp.unique(labels).astype(cp.int32)
        valid_clusters = valid_clusters[valid_clusters >= 0]
        num_valid = len(valid_clusters)
        if num_valid == 0: return

        valid_mask_v = (labels >= 0)
        active_vars = cp.where(valid_mask_v)[0]
        active_labels = labels[valid_mask_v]
        if len(active_vars) == 0: return
        data_v = cp.ones(len(active_vars), dtype=cp.bool_)
        cluster_to_vars = cpx.coo_matrix((data_v, (active_labels, active_vars)), shape=(n_comps, self.N)).tocsr()

        flat_clusters = lit_clusters.flatten()
        flat_clauses = cp.repeat(cp.arange(self.M), 3)
        mask_c = (flat_clusters >= 0)
        flat_clusters = flat_clusters[mask_c]
        flat_clauses = flat_clauses[mask_c]
        if len(flat_clusters) == 0: return
        combined_keys = flat_clusters.astype(cp.int64) * self.M + flat_clauses.astype(cp.int64)
        unique_keys = cp.unique(combined_keys)
        u_clusters = (unique_keys // self.M).astype(cp.int32)
        u_clauses = (unique_keys % self.M).astype(cp.int32)
        data_c = cp.ones(len(u_clusters), dtype=cp.bool_)
        cluster_to_clauses = cpx.coo_matrix((data_c, (u_clusters, u_clauses)), shape=(n_comps, self.M)).tocsr()

        c2c_indptr = cluster_to_clauses.indptr.astype(cp.int32)
        c2c_indices = cluster_to_clauses.indices.astype(cp.int32)
        c2v_indptr = cluster_to_vars.indptr.astype(cp.int32)
        c2v_indices = cluster_to_vars.indices.astype(cp.int32)
        lit_clusters_ptr = cp.ascontiguousarray(lit_clusters.astype(cp.int32))
        seed = int(time.time() * 1000) % 1000000007

        self.kernel(
            (1,), (256,),
            (
                self.sigma, c2c_indptr, c2c_indices, c2v_indptr, c2v_indices,
                self.lits_idx, self.lits_sign, lit_clusters_ptr, valid_clusters,
                cp.int32(num_valid), cp.int32(self.steps_flips),
                cp.float32(omega), cp.float32(self.beta_scale), cp.uint64(seed),
                cp.int32(require_unsat)
            )
        )

    def step(self, omega, verbose=False):
        c_spins = self.sigma[self.lits_idx]
        s0, s1, s2 = c_spins[:, 0], c_spins[:, 1], c_spins[:, 2]
        sat0 = (s0 * s1 * self.J_tri[:, 0] == 1)
        sat1 = (s1 * s2 * self.J_tri[:, 1] == 1)
        sat2 = (s2 * s0 * self.J_tri[:, 2] == 1)
        sat_mask = cp.stack([sat0, sat1, sat2], axis=1)
        num_sat_tri = cp.sum(sat_mask, axis=1)
        is_low_energy = (num_sat_tri == 2)

        lit_is_sat = (c_spins == self.lits_sign)
        is_unsat = (cp.sum(lit_is_sat, axis=1) == 0)
        
        # Mark variables involved in UNSAT clauses
        marked_vars = cp.zeros(self.N, dtype=bool)
        if cp.any(is_unsat):
            unsat_indices = self.lits_idx[is_unsat].flatten()
            marked_vars[unsat_indices] = True
        
        lit_marked = marked_vars[self.lits_idx] # (M, 3)
        num_marked_in_clause = cp.sum(lit_marked, axis=1) # (M,)

        P = 1.0 - cp.exp(-omega)
        rand_vals = cp.random.random(self.M, dtype=cp.float32)
        src_nodes, dst_nodes = [], []

        # Mask 1: Clean (Low Energy AND No Marked Vars)
        mask_clean = is_low_energy & (num_marked_in_clause == 0) & (rand_vals < P)
        
        if cp.any(mask_clean):
            idx_C = cp.where(mask_clean)[0]
            # Standard P/2 logic for clean triangles
            r_vals_C = rand_vals[idx_C]
            sub_sat = sat_mask[idx_C]
            idx_1st = cp.argmax(sub_sat, axis=1)
            idx_sum = cp.sum(sub_sat * cp.array([0, 1, 2], dtype=cp.int8), axis=1)
            idx_2nd = idx_sum - idx_1st
            pick_first = (r_vals_C < (P / 2.0))
            chosen = cp.where(pick_first, idx_1st, idx_2nd)
            
            lits = self.lits_idx[idx_C]
            l0, l1, l2 = lits[:,0], lits[:,1], lits[:,2]
            s_e = cp.where(chosen==0, l0, cp.where(chosen==1, l1, l2))
            d_e = cp.where(chosen==0, l1, cp.where(chosen==1, l2, l0))
            src_nodes.append(s_e)
            dst_nodes.append(d_e)

        # Mask 2: Dirty (Low Energy AND Marked Vars)
        mask_dirty = is_low_energy & (num_marked_in_clause > 0)
        if cp.any(mask_dirty):
            idx_D = cp.where(mask_dirty)[0]
            r_vals_D = rand_vals[idx_D]
            
            # Opposites:
            # Edge 0 (0-1) is opposite to vertex 2
            # Edge 1 (1-2) is opposite to vertex 0
            # Edge 2 (2-0) is opposite to vertex 1
            
            # Check Edge 0: is sat0 AND vertex 2 marked?
            edge0_active = sat_mask[idx_D, 0] & lit_marked[idx_D, 2] & (r_vals_D < P)
            if cp.any(edge0_active):
                sub = idx_D[edge0_active]
                l = self.lits_idx[sub]
                src_nodes.append(l[:,0])
                dst_nodes.append(l[:,1])
                
            # Check Edge 1: is sat1 AND vertex 0 marked?
            edge1_active = sat_mask[idx_D, 1] & lit_marked[idx_D, 0] & (r_vals_D < P)
            if cp.any(edge1_active):
                sub = idx_D[edge1_active]
                l = self.lits_idx[sub]
                src_nodes.append(l[:,1])
                dst_nodes.append(l[:,2])
                
            # Check Edge 2: is sat2 AND vertex 1 marked?
            edge2_active = sat_mask[idx_D, 2] & lit_marked[idx_D, 1] & (r_vals_D < P)
            if cp.any(edge2_active):
                sub = idx_D[edge2_active]
                l = self.lits_idx[sub]
                src_nodes.append(l[:,2])
                dst_nodes.append(l[:,0])

        if len(src_nodes) > 0:
            all_src = cp.concatenate(src_nodes)
            all_dst = cp.concatenate(dst_nodes)
            data = cp.ones(len(all_src), dtype=cp.float32)
            adj = cpx.coo_matrix((data, (all_src, all_dst)), shape=(self.N, self.N))
            n_comps, labels = cpx_graph.connected_components(adj, directed=False)
        else:
            n_comps = self.N
            labels = cp.arange(self.N, dtype=cp.int32)

        # ER Super Clustering
        m = n_comps
        final_labels = labels
        final_n_comps = n_comps
        if m > 1 and self.a > 0:
            num_edges = int(self.a * (m - 1) / 2)
            if num_edges > 0:
                s_er = cp.random.randint(0, m, size=num_edges, dtype=cp.int32)
                d_er = cp.random.randint(0, m, size=num_edges, dtype=cp.int32)
                data_er = cp.ones(num_edges, dtype=cp.float32)
                adj_er = cpx.coo_matrix((data_er, (s_er, d_er)), shape=(m, m))
                n_super, super_labels = cpx_graph.connected_components(adj_er, directed=False)
                final_labels = super_labels[labels]
                final_n_comps = n_super

        if verbose:
            print(f"Phase 1: {n_comps} clusters -> {final_n_comps} super. Clean/Dirty logic.")

        # 2. Dynamics Phase 1: GLOBAL (require_unsat = 0)
        self._run_dynamics(final_labels, final_n_comps, omega, require_unsat=0)

        # 3. Dynamics Phase 2: LOCAL UNSAT (require_unsat = 1)
        # Only UNSAT triangles
        
        # Re-eval UNSAT status
        c_spins = self.sigma[self.lits_idx]
        lit_is_sat = (c_spins == self.lits_sign)
        is_unsat = (cp.sum(lit_is_sat, axis=1) == 0)
        
        if cp.any(is_unsat):
            omega_2 = 8.0 * omega
            P_2 = 1.0 - cp.exp(-omega_2)
            P_7 = P_2 / 7.0 # 2P/7 for each edge
            T_freeze = 2.0 * P_7 
            
            idx_U = cp.where(is_unsat)[0]
            n_unsat = len(idx_U)
            r_vals_U = cp.random.random(n_unsat, dtype=cp.float32)
            src_2, dst_2 = [], []
            
            # Edge 0
            mask_e0 = (r_vals_U < T_freeze)
            if cp.any(mask_e0):
                l = self.lits_idx[idx_U[mask_e0]]
                src_2.append(l[:,0]); dst_2.append(l[:,1])
            
            # Edge 1 (Independent probability? Prompt implies mutually exclusive "prob 2P/7 gèle 1ere, prob 2P/7 gèle 2eme...")
            # If so, ranges are [0, 2/7), [2/7, 4/7), [4/7, 6/7).
            
            mask_e1 = (r_vals_U >= T_freeze) & (r_vals_U < 2.0 * T_freeze)
            if cp.any(mask_e1):
                l = self.lits_idx[idx_U[mask_e1]]
                src_2.append(l[:,1]); dst_2.append(l[:,2])
                
            mask_e2 = (r_vals_U >= 2.0 * T_freeze) & (r_vals_U < 3.0 * T_freeze)
            if cp.any(mask_e2):
                l = self.lits_idx[idx_U[mask_e2]]
                src_2.append(l[:,2]); dst_2.append(l[:,0])
                
            if len(src_2) > 0:
                all_src_2 = cp.concatenate(src_2)
                all_dst_2 = cp.concatenate(dst_2)
                data_2 = cp.ones(len(all_src_2), dtype=cp.float32)
                adj_2 = cpx.coo_matrix((data_2, (all_src_2, all_dst_2)), shape=(self.N, self.N))
                n_comps_2, labels_2 = cpx_graph.connected_components(adj_2, directed=False)
            else:
                n_comps_2 = self.N
                labels_2 = cp.arange(self.N, dtype=cp.int32)
            
            # ER Super Clustering on Phase 2
            m = n_comps_2
            final_labels_2 = labels_2
            final_n_comps_2 = n_comps_2
            
            if m > 1 and self.a > 0:
                num_edges = int(self.a * (m - 1) / 2)
                if num_edges > 0:
                    s_er = cp.random.randint(0, m, size=num_edges, dtype=cp.int32)
                    d_er = cp.random.randint(0, m, size=num_edges, dtype=cp.int32)
                    data_er = cp.ones(num_edges, dtype=cp.float32)
                    adj_er = cpx.coo_matrix((data_er, (s_er, d_er)), shape=(m, m))
                    n_super, super_labels = cpx_graph.connected_components(adj_er, directed=False)
                    final_labels_2 = super_labels[labels_2]
                    final_n_comps_2 = n_super
            
            if verbose:
                print(f"Phase 2 (UNSAT): {n_comps_2} clusters -> {final_n_comps_2} super.")

            self._run_dynamics(final_labels_2, final_n_comps_2, omega_2, require_unsat=1)

        e = self.energy_check()
        if e < self.min_energy:
            self.min_energy = e
            self.best_sigma = self.sigma.copy()
            if e == 0.0: print("SOLUTION FOUND!")
        
        return e, 0.0, 0.0

class WalkSAT:
    def __init__(self, clauses_np, N):
        self.N = N
        self.clauses = clauses_np 
        self.M = len(clauses_np)
        self.sigma = np.random.choice([-1, 1], size=N+1)
        self.sigma[0] = 1

    def evaluate(self):
        lits = self.clauses
        s = self.sigma[np.abs(lits)]
        sat = (lits * s) > 0
        clause_sat = np.any(sat, axis=1)
        return np.where(~clause_sat)[0], 1.0 - np.mean(clause_sat)

    def step(self, flips=1):
        unsat_indices, _ = self.evaluate()
        if len(unsat_indices) == 0: return 0.0
        for _ in range(flips):
            if len(unsat_indices) == 0: break
            target_clause = np.random.choice(unsat_indices)
            vars_c = np.abs(self.clauses[target_clause])
            v_flip = np.random.choice(vars_c)
            self.sigma[v_flip] *= -1
        _, e = self.evaluate()
        return e

# --- MAIN ---

if __name__ == "__main__":
    N = 10000
    alpha = 4.0 
    clauses_np, _ = generate_random_3sat(N, alpha, seed=42)
    print(f"Instance: N={N}, M={len(clauses_np)}, Alpha={alpha}")

    # Solvers
    solver_er = SwendsenWangErdosRenyiGPU(clauses_np, N, beta_scale=100.0, steps_flips=2*N, a=0.9)
    solver_constrained = ConstrainedSwendsenWangErdosRenyiGPU(clauses_np, N, beta_scale=100.0, steps_flips=2*N, a=0.9)
    solver_complete_er = SwendsenWangCompleteErdosRenyiGPU(clauses_np, N, beta_scale=100.0, steps_flips=2*N, a=0.9)
    solver_gl = SwendsenWangGlauberGPU(clauses_np, N, beta_scale=100.0, steps_flips=2*N)
    solver_complete = CompleteSwendsenWangGPU(clauses_np, N, beta_scale=100.0, steps_flips=2*N)
    solver_dyn = DynamicsUNSAT_GPU(clauses_np, N, beta_scale=100.0, steps_flips=2*N, a=0.9)
    walksat = WalkSAT(clauses_np, N)

    steps = 10000
    omega_min = 0.05
    omega_max = 0.2
    epsilon = 1e-4
    raw_decay = np.geomspace(1, epsilon, steps)
    decay_01 = (raw_decay - epsilon) / (1.0 - epsilon)
    omega_schedule = omega_max - (omega_max - omega_min) * decay_01

    history_er = []
    history_const = []
    history_c_er = []
    history_gl = []
    history_cp = []
    history_ws = []
    history_dyn = []

    t0 = time.time()
    print("Starting Comparison...")

    for i, omega in enumerate(omega_schedule):
        is_verbose = (i % 50 == 0)
        
        unsat_er, _, _ = solver_er.step(omega, verbose=False)
        history_er.append(float(unsat_er.get()) if hasattr(unsat_er, 'get') else float(unsat_er))

        unsat_const, _, _ = solver_constrained.step(omega, verbose=False)
        history_const.append(float(unsat_const.get()) if hasattr(unsat_const, 'get') else float(unsat_const))

        unsat_c_er, _, _ = solver_complete_er.step(omega, verbose=is_verbose)
        history_c_er.append(float(unsat_c_er.get()) if hasattr(unsat_c_er, 'get') else float(unsat_c_er))

        unsat_gl, _, _ = solver_gl.step(omega, verbose=False)
        history_gl.append(float(unsat_gl.get()) if hasattr(unsat_gl, 'get') else float(unsat_gl))
        
        unsat_cp, _, _ = solver_complete.step(omega, verbose=False)
        history_cp.append(float(unsat_cp.get()) if hasattr(unsat_cp, 'get') else float(unsat_cp))

        unsat_dyn, _, _ = solver_dyn.step(omega, verbose=is_verbose)
        history_dyn.append(float(unsat_dyn.get()) if hasattr(unsat_dyn, 'get') else float(unsat_dyn))

        flips_per_step = N // 10000
        if flips_per_step < 1: flips_per_step = 1
        e_ws = 1.0
        for _ in range(flips_per_step):
            e_ws = walksat.step(flips=1)
            if e_ws == 0.0: break
        history_ws.append(e_ws)

        if is_verbose:
            print(f"Step {i:4d} | w={omega:.3f} | ER: {unsat_er:.5f} | CST: {unsat_const:.5f} | C-ER: {unsat_c_er:.5f} | DYN: {unsat_dyn:.5f} | GL: {unsat_gl:.5f} | CP: {unsat_cp:.5f} | WS: {e_ws:.5f}")
            if unsat_c_er == 0.0: print("COMPLETE-ER SOLVED!"); break
            if unsat_const == 0.0: print("CONST SOLVED!"); break
            if unsat_er == 0.0: print("ER SOLVED!"); break
            if unsat_dyn == 0.0: print("DYN SOLVED!"); break

    dt = time.time() - t0
    print(f"Done in {dt:.2f}s")

    # Plot
    omega_cpu = omega_schedule[:len(history_er)]
    er_cpu = np.array(history_er)
    const_cpu = np.array(history_const)
    c_er_cpu = np.array(history_c_er)
    gl_cpu = np.array(history_gl)
    cp_cpu = np.array(history_cp)
    ws_cpu = np.array(history_ws)
    dyn_cpu = np.array(history_dyn)

    plt.figure(figsize=(12, 7))
    ax1 = plt.gca()

    l1, = ax1.plot(omega_cpu, er_cpu, label='Erdos-Renyi SW', color='cyan', linewidth=1.5, alpha=0.5)
    l2, = ax1.plot(omega_cpu, const_cpu, label='Constrained ER-SW', color='magenta', linewidth=1.5, alpha=0.5)
    l3, = ax1.plot(omega_cpu, c_er_cpu, label='Complete ER-SW (New)', color='white', linewidth=2.5)
    l4, = ax1.plot(omega_cpu, gl_cpu, label='Glauber', color='lime', linewidth=1.5, linestyle=':', alpha=0.5)
    l5, = ax1.plot(omega_cpu, cp_cpu, label='Complete', color='yellow', linewidth=1.5, linestyle=':', alpha=0.5)
    l6, = ax1.plot(omega_cpu, ws_cpu, label='WalkSAT', color='red', alpha=0.5)
    l7, = ax1.plot(omega_cpu, dyn_cpu, label='Dynamics UNSAT', color='orange', linewidth=2.0)

    ax1.set_xlabel(r'Coupling $\omega$ (Time)')
    ax1.set_ylabel('Fraction Unsatisfied', color='white')
    ax1.tick_params(axis='y', labelcolor='white')
    ax1.grid(True, alpha=0.2)

    ax1.legend(loc='upper right')
    plt.title(f'Solver Comparison (N={N}, Alpha={alpha})')
    plt.savefig('comparison_plot.png')