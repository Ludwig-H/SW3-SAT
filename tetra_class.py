# @title 3. The Solver: `TetraDynamicsGPU`
# Implements the Generalized Higher-Order Cluster Dynamics with Optimal Energy Transfer (LP).

import scipy.sparse as sp
from scipy.optimize import linprog

class TetraDynamicsGPU:
    def __init__(self, clauses_np, N, omega=2.0):
        """
        Initialize the Generalized Higher-Order Cluster Solver.
        clauses_np: (M, 3) numpy array of literals (int32).
        N: Number of variables.
        omega: Energy scaling parameter.
        """
        self.N = N
        self.raw_clauses = clauses_np # For global energy check
        self.omega = omega
        
        # --- 1. Topology Builder: Find All Candidates & Solve LP ---
        print("Decomposing graph topology and optimizing energy transfer...")
        tetras, triangles = self._build_topology_optimized(clauses_np)
        print(f"Topology: {len(tetras)} Active Tetrahedrons, {len(triangles)} Residual Triangles.")
        
        # --- 2. Prepare Data for GPU (Tetrahedrons) ---
        self.num_tetras = len(tetras)
        if self.num_tetras > 0:
            t_indices = np.array([t['indices'] for t in tetras], dtype=np.int32)
            t_signs   = np.array([t['signs'] for t in tetras], dtype=np.int8)
            t_active  = np.array([t['active'] for t in tetras], dtype=bool)
            t_m       = np.array([t['m'] for t in tetras], dtype=np.int8)
            t_weights = np.array([t['weight'] for t in tetras], dtype=np.float32)
            
            self.t_indices = cp.array(t_indices)
            self.t_signs   = cp.array(t_signs)
            self.t_active  = cp.array(t_active) # Mask: True if node is in Active Set A
            self.t_m       = cp.array(t_m)
            self.t_weights = cp.array(t_weights)
        else:
            self.t_indices = cp.empty((0, 4), dtype=cp.int32)
            self.t_weights = cp.empty((0,), dtype=cp.float32)
            self.t_signs   = cp.empty((0, 4), dtype=cp.int8)
            self.t_active  = cp.empty((0, 4), dtype=bool)
            self.t_m       = cp.empty((0,), dtype=np.int8)
            
        # --- 3. Prepare Data for GPU (Residual Triangles) ---
        self.num_tris = len(triangles)
        if self.num_tris > 0:
            r_indices = np.array([t['indices'] for t in triangles], dtype=np.int32)
            r_signs   = np.array([t['signs'] for t in triangles], dtype=np.int8)
            r_weights = np.array([t['weight'] for t in triangles], dtype=np.float32)
            
            self.r_indices = cp.array(r_indices)
            self.r_signs   = cp.array(r_signs)
            self.r_weights = cp.array(r_weights)
        else:
            self.r_indices = cp.empty((0, 3), dtype=cp.int32)
            self.r_weights = cp.empty((0,), dtype=cp.float32)
            self.r_signs   = cp.empty((0, 3), dtype=cp.int8)

        # Initialize Spins
        self.spins = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=N + 1)
        
        # Ghost Node Indices
        self.GHOST_PLUS = N + 1
        self.GHOST_MINUS = N + 2
        self.TOTAL_NODES = N + 3
        
        # Constants are now per-object, so we don't precompute global exp_w here.

    def _build_topology_optimized(self, clauses):
        """
        1. Identifies ALL possible tetrahedrons (m=2,3,4).
        2. Solves LP to assign weights w_t such that sum(w_t) <= omega for each clause.
        3. Returns active tetrahedrons (w_t > 0) and residual triangles.
        """
        from collections import defaultdict
        import itertools
        
        clause_sets = [tuple(sorted(c)) for c in clauses]
        clause_to_id = {c: i for i, c in enumerate(clause_sets)}
        num_clauses = len(clauses)
        
        # --- A. Find Candidates ---
        # Map Edge -> Clauses
        edge_map = defaultdict(list)
        for idx, literals in enumerate(clause_sets):
            l = literals
            edge_map[tuple(sorted((l[0], l[1])))].append(idx)
            edge_map[tuple(sorted((l[0], l[2])))].append(idx)
            edge_map[tuple(sorted((l[1], l[2])))].append(idx)
            
        seen_tetra = set()
        candidates = [] # List of dicts
        
        # Helper to build tetra data
        def build_candidate(idx1, idx2):
            l1 = set(clause_sets[idx1])
            l2 = set(clause_sets[idx2])
            union_l = sorted(list(l1 | l2))
            
            if len(union_l) != 4: return None
            
            t_key = tuple(union_l)
            if t_key in seen_tetra: return None
            seen_tetra.add(t_key)
            
            # Identify Faces
            faces = list(itertools.combinations(union_l, 3))
            found_clause_indices = []
            
            active_mask = [False] * 4
            # Map union_l to 0..3
            
            for i in range(4):
                # Face opposite to i
                face_lits = tuple(sorted([union_l[k] for k in range(4) if k != i]))
                if face_lits in clause_to_id:
                    cid = clause_to_id[face_lits]
                    found_clause_indices.append(cid)
                    active_mask[i] = True
            
            m = len(found_clause_indices)
            if m < 2: return None
            
            return {
                'literals': union_l,
                'clauses': found_clause_indices,
                'active': active_mask,
                'm': m
            }

        # Iterate edges
        for edge, c_indices in edge_map.items():
            if len(c_indices) < 2: continue
            # Check all pairs
            for i in range(len(c_indices)):
                for j in range(i+1, len(c_indices)):
                    cand = build_candidate(c_indices[i], c_indices[j])
                    if cand:
                        candidates.append(cand)
                        
        print(f"LP Optimization: Found {len(candidates)} candidate tetrahedrons.")
        
        # --- B. Linear Programming ---
        # Maximize sum(w_t)
        # Subject to: A * w <= omega
        # w >= 0
        
        if len(candidates) == 0:
            # Fallback: No tetras, all triangles
            residual_tris = []
            for i, c in enumerate(clause_sets):
                residual_tris.append({
                    'indices': [abs(x)-1 for x in c],
                    'signs': [int(np.sign(x)) for x in c],
                    'weight': self.omega
                })
            return [], residual_tris

        # Build Sparse Matrix A
        # Rows: Clauses, Cols: Candidates
        # Value 1 if candidate uses clause
        
        row_ind = []
        col_ind = []
        data = []
        
        for t_idx, cand in enumerate(candidates):
            for c_idx in cand['clauses']:
                row_ind.append(c_idx)
                col_ind.append(t_idx)
                data.append(1.0)
                
        A = sp.csr_matrix((data, (row_ind, col_ind)), shape=(num_clauses, len(candidates)))
        
        # Objective: Maximize sum(w), so minimize -sum(w)
        c_obj = -1.0 * np.ones(len(candidates))
        
        # Constraints: A * w <= omega
        b_ub = np.full(num_clauses, self.omega)
        
        print("Solving Linear Program...")
        res = linprog(c_obj, A_ub=A, b_ub=b_ub, bounds=(0, None), method='highs')
        
        if not res.success:
            print(f"LP Warning: {res.message}")
            w_t = np.zeros(len(candidates)) # Fail safe
        else:
            w_t = res.x
        
        # --- C. Reconstruct Objects ---
        final_tetras = []
        
        # Determine consumed weights per clause
        # clause_consumption[c] = sum(w_t for t containing c)
        clause_consumption = A.dot(w_t)
        
        # Threshold to consider a tetra active
        EPSILON = 1e-5
        
        for i, cand in enumerate(candidates):
            weight = w_t[i]
            if weight > EPSILON:
                # Build Tetra Object
                lits = cand['literals']
                indices = [abs(x)-1 for x in lits]
                signs = [int(np.sign(x)) for x in lits]
                
                final_tetras.append({
                    'indices': indices,
                    'signs': signs,
                    'active': cand['active'],
                    'm': cand['m'],
                    'weight': weight
                })
                
        # Residual Triangles
        final_tris = []
        for i in range(num_clauses):
            remaining = self.omega - clause_consumption[i]
            if remaining > EPSILON:
                c = clause_sets[i]
                final_tris.append({
                    'indices': [abs(x)-1 for x in c],
                    'signs': [int(np.sign(x)) for x in c],
                    'weight': remaining
                })
                
        return final_tetras, final_tris

    def step(self):
        """
        Swendsen-Wang Step with Generalized Dynamics and Heterogeneous Weights.
        """
        # --- 1. Tetrahedrons Dynamics ---
        if self.num_tetras > 0:
            # Gather spins (T, 4)
            t_spins = self.spins[self.t_indices]
            # Check sat (T, 4)
            is_sat = (t_spins == self.t_signs)
            # k: num sat (T,)
            k = cp.sum(is_sat, axis=1)
            
            # --- Determine State & Probabilities ---
            
            # We use unified random u
            u = cp.random.random(self.num_tetras, dtype=cp.float32)
            bonds = cp.zeros(self.num_tetras, dtype=cp.int8)
            
            # Masks
            mask_k1 = (k == 1)
            mask_k1_active = mask_k1 & cp.any(is_sat & self.t_active, axis=1)
            mask_k1_inactive = mask_k1 & (~mask_k1_active)
            mask_k_ge_2 = (k >= 2)
            
            state = cp.zeros(self.num_tetras, dtype=cp.int8)
            state[mask_k1_active] = 1
            state[mask_k1_inactive] = 2 # Equivalent to k>=2 for energy purposes
            state[mask_k_ge_2] = 2
            
            # We compute probabilities based on m and w_t.
            # We need arrays for m.
            m_arr = self.t_m
            w_arr = self.t_weights
            
            # State 0 (k=0): P(B=0) = 1.0 (Always 0, never freeze)
            
            # State 1 (k=1 Active):
            # Freezing 1 bond is preferred.
            # Threshold T1 = exp( - (m - 1) * w )
            # if u < T1: B=0. Else B=1.
            p_b0_s1 = cp.exp(- (m_arr - 1.0) * w_arr)
            
            mask_S1 = (state == 1)
            mask_S1_B1 = mask_S1 & (u >= p_b0_s1)
            bonds[mask_S1_B1] = 1
            
            # State 2 (k>=2 or k=1 Inactive):
            # Can freeze 1 or 2 bonds.
            # T1 (B=0) = exp( - m * w )
            # T2 (B<=1) = exp( - w )
            
            p_b0_s2 = cp.exp(- m_arr * w_arr)
            p_b1_cum_s2 = cp.exp(-1.0 * w_arr)
            
            mask_S2 = (state == 2)
            mask_S2_B1 = mask_S2 & (u >= p_b0_s2) & (u < p_b1_cum_s2)
            bonds[mask_S2_B1] = 1
            
            mask_S2_B2 = mask_S2 & (u >= p_b1_cum_s2)
            bonds[mask_S2_B2] = 2
            
            # --- Witness Selection (Standard) ---
            # This part depends on geometry/satisfaction, not weights.
            priorities = cp.random.random((self.num_tetras, 4), dtype=cp.float32)
            w1_global = cp.full(self.num_tetras, -1, dtype=cp.int32)
            w2_global = cp.full(self.num_tetras, -1, dtype=cp.int32)
            w1_sign   = cp.zeros(self.num_tetras, dtype=cp.int8)
            w2_sign   = cp.zeros(self.num_tetras, dtype=cp.int8)
            
            # B=1 Selection
            mask_active_1 = (bonds >= 1)
            p_sat = priorities.copy()
            p_sat[~is_sat] = -2.0
            idx_w1 = cp.argmax(p_sat, axis=1)
            
            w1_global = cp.where(mask_active_1, 
                                 cp.take_along_axis(self.t_indices, idx_w1[:, None], axis=1).flatten(),
                                 w1_global)
            w1_sign = cp.where(mask_active_1,
                               cp.take_along_axis(self.t_signs, idx_w1[:, None], axis=1).flatten(),
                               w1_sign)
            
            # B=2 Selection
            mask_B2 = (bonds == 2)
            if cp.any(mask_B2):
                has_sat_inactive = cp.any(is_sat & (~self.t_active), axis=1)
                mask_B2_Inactive = mask_B2 & has_sat_inactive
                mask_B2_ActiveOnly = mask_B2 & (~has_sat_inactive)
                
                # Inactive Priority
                p_sat_inactive = p_sat.copy()
                p_sat_inactive[self.t_active] = -2.0
                idx_w1_inactive = cp.argmax(p_sat_inactive, axis=1)
                
                w1_global = cp.where(mask_B2_Inactive, 
                                     cp.take_along_axis(self.t_indices, idx_w1_inactive[:,None], axis=1).flatten(),
                                     w1_global)
                w1_sign = cp.where(mask_B2_Inactive,
                                   cp.take_along_axis(self.t_signs, idx_w1_inactive[:,None], axis=1).flatten(),
                                   w1_sign)
                
                # Active Only Priority (Pick 2nd witness)
                p_sat_w2 = p_sat.copy()
                rows = cp.arange(self.num_tetras)
                idx_w1_current = cp.argmax(p_sat, axis=1)
                p_sat_w2[rows, idx_w1_current] = -3.0
                idx_w2 = cp.argmax(p_sat_w2, axis=1)
                
                w2_global = cp.where(mask_B2_ActiveOnly,
                                     cp.take_along_axis(self.t_indices, idx_w2[:,None], axis=1).flatten(),
                                     w2_global)
                w2_sign = cp.where(mask_B2_ActiveOnly,
                                   cp.take_along_axis(self.t_signs, idx_w2[:,None], axis=1).flatten(),
                                   w2_sign)

            mask_has_w1 = (bonds >= 1)
            mask_has_w2 = (w2_global != -1)
            
            src_t1 = w1_global[mask_has_w1]
            tgt_t1 = cp.where(w1_sign[mask_has_w1] > 0, self.GHOST_PLUS, self.GHOST_MINUS)
            
            src_t2 = w2_global[mask_has_w2]
            tgt_t2 = cp.where(w2_sign[mask_has_w2] > 0, self.GHOST_PLUS, self.GHOST_MINUS)
        
        else:
            src_t1, tgt_t1 = cp.array([], dtype=cp.int32), cp.array([], dtype=cp.int32)
            src_t2, tgt_t2 = cp.array([], dtype=cp.int32), cp.array([], dtype=cp.int32)


        # --- 2. Residual Triangles Dynamics ---
        if self.num_tris > 0:
            # Gather spins
            r_spins = self.spins[self.r_indices]
            # Sat
            r_is_sat = (r_spins == self.r_signs)
            r_clause_sat = cp.any(r_is_sat, axis=1)
            
            # Sampling: Only if satisfied
            # P(Freeze) = 1 - e^-w_r
            p_freeze = 1.0 - cp.exp(-self.r_weights)
            
            u_r = cp.random.random(self.num_tris, dtype=cp.float32)
            
            mask_freeze = r_clause_sat & (u_r < p_freeze)
            
            # Select 1 witness (Random Sat)
            r_priorities = cp.random.random((self.num_tris, 3), dtype=cp.float32)
            r_priorities[~r_is_sat] = -1.0
            
            idx_r_w1 = cp.argmax(r_priorities, axis=1)
            
            # Extract
            src_r = cp.take_along_axis(self.r_indices, idx_r_w1[:,None], axis=1).flatten()[mask_freeze]
            sign_r = cp.take_along_axis(self.r_signs, idx_r_w1[:,None], axis=1).flatten()[mask_freeze]
            tgt_r = cp.where(sign_r > 0, self.GHOST_PLUS, self.GHOST_MINUS)
            
        else:
            src_r, tgt_r = cp.array([], dtype=cp.int32), cp.array([], dtype=cp.int32)
            
        # --- 3. Graph Construction & Cluster Flip ---
        all_src = cp.concatenate([src_t1, src_t2, src_r])
        all_tgt = cp.concatenate([tgt_t1, tgt_t2, tgt_r])
        
        if len(all_src) > 0:
            weights = cp.ones(len(all_src), dtype=cp.float32)
            # Create CSR Matrix
            adj = cpx.coo_matrix((weights, (all_src, all_tgt)), shape=(self.TOTAL_NODES, self.TOTAL_NODES), dtype=cp.float32)
            adj = adj.tocsr()
            adj = adj + adj.T
            
            # Components
            n_components, labels = cpx_graph.connected_components(adj, directed=False)
            
            # Determine Ghost Labels
            l_plus = labels[self.GHOST_PLUS]
            l_minus = labels[self.GHOST_MINUS]
            
            # Random Flips
            comp_flips = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=n_components)
            
            new_spins = comp_flips[labels[:self.N+1]]
            
            # Fix Ghosts
            mask_plus = (labels[:self.N+1] == l_plus)
            mask_minus = (labels[:self.N+1] == l_minus)
            
            new_spins[mask_plus] = 1
            new_spins[mask_minus] = -1
            
            # Conflict handling (Rare)
            mask_conflict = mask_plus & mask_minus
            if cp.any(mask_conflict):
                new_spins[mask_conflict] = self.spins[mask_conflict] # Freeze to old
                
            self.spins = new_spins
        else:
            # Fallback for empty edges (Global random flip)
            self.spins = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=self.N + 1)

    def energy(self):
        """Global 3-SAT Energy (Fraction Unsatisfied Clauses)."""
        # (M, 3)
        indices = cp.array(self.raw_clauses - 1) # 0-based
        signs = cp.array(np.sign(self.raw_clauses), dtype=cp.int8)
        indices = cp.abs(indices)
        
        current_spins = self.spins[indices]
        is_sat = (current_spins == signs)
        clause_sat = cp.any(is_sat, axis=1)
        
        return 1.0 - cp.mean(clause_sat)
