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
# We check for GPU availability and ensure the CuPy version matches the Driver.

import sys
import os
import subprocess
import time
import warnings
import requests

# Function to force install a compatible CuPy
def install_compatible_cupy():
    print("Installing cupy-cuda11x (broad compatibility)...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'cupy', 'cupy-cuda12x', 'cupy-cuda11x'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'cupy-cuda11x'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
        return

    print("Installation complete.")
    print("⚠️ CRITICAL: The runtime will now RESTART automatically to load the new library.")
    print("⚠️ You may see a 'Session Crashed' or 'Kernel Restarting' message. This is NORMAL.")
    print("⚠️ AFTER the restart, please RE-RUN this cell manually.")
    time.sleep(2)
    # Kill the current process to force Colab/Jupyter to restart the kernel
    os.kill(os.getpid(), 9)

try:
    import cupy as cp
    # aggressive check: try to allocate and execute a small kernel
    x = cp.array([1.0, 2.0])
    y = x * x
    print(f"GPU Detected: {cp.cuda.runtime.getDeviceCount()} device(s)")
    print(f"CuPy Version: {cp.__version__}")
except ImportError:
    print("CuPy not installed.")
    install_compatible_cupy()
except Exception as e:
    # Catch CUDARuntimeError or generic exceptions related to driver mismatch
    print(f"GPU/CuPy check failed: {e}")
    if "InsufficientDriver" in str(e) or "cudaErrorInsufficientDriver" in str(e):
        print("Driver is too old for the installed CuPy runtime.")
    install_compatible_cupy()

import numpy as np
import matplotlib.pyplot as plt
import cupyx.scipy.sparse as cpx
import cupyx.scipy.sparse.csgraph as cpx_graph

# Graphics settings
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (12, 6)
"""
b.cells.append(nbf.v4.new_code_cell(code_setup))

# Cell 3: Data Gen
code_data = r'''# @title 2. Data Generation & Parsing

import gzip
import io
import tarfile

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

def parse_dimacs(content_str):
    """Parses DIMACS CNF content string."""
    clauses = []
    N = 0
    for line in content_str.splitlines():
        line = line.strip()
        if not line or line.startswith('c') or line.startswith('%'): continue
        if line.startswith('p'):
            parts = line.split()
            try:
                N = int(parts[2])
            except:
                pass # sometimes header is malformed
            continue
        
        # Parse literals
        try:
            lits = [int(x) for x in line.split() if x != '0']
            if len(lits) >= 3:
                # We take the first 3 literals for 3-SAT (truncating if >3, though ideal is proper 3-SAT)
                # Or skip if not 3-SAT? For now, we assume input is 3-SAT.
                # If length < 3, we might need padding.
                # Let's strictly take triplets or skip.
                if len(lits) == 3:
                    clauses.append(lits)
        except ValueError:
            continue
            
    # Auto-detect N if header failed
    clauses_np = np.array(clauses, dtype=np.int32)
    if N == 0 and len(clauses_np) > 0:
        N = np.max(np.abs(clauses_np))
        
    return clauses_np, N

def download_and_parse_instance(url):
    """Downloads and parses a CNF instance (supports .cnf, .cnf.gz, .tar.gz)."""
    print(f"Downloading {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        print(f"Download Error: {e}")
        return np.array([]), 0
    
    content = response.content
    text_content = None
    
    # 1. Check for tar.gz
    if url.endswith('.tar.gz') or url.endswith('.tgz'):
        try:
            with tarfile.open(fileobj=io.BytesIO(content), mode='r:gz') as tar:
                # Find first .cnf file
                for member in tar.getmembers():
                    if member.name.endswith('.cnf'):
                        print(f"Extracting {member.name} from archive...")
                        f = tar.extractfile(member)
                        if f:
                            text_content = f.read().decode('utf-8', errors='ignore')
                            break
                if text_content is None:
                    print("No .cnf file found in archive.")
                    return np.array([]), 0
        except Exception as e:
            print(f"Error extracting tar.gz: {e}")
            return np.array([]), 0
            
    # 2. Check for .gz (single file)
    elif url.endswith('.gz'):
        try:
            with gzip.open(io.BytesIO(content), 'rt', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()
        except Exception as e:
            print(f"Error decompressing .gz: {e}")
            return np.array([]), 0
            
    # 3. Plain text
    else:
        text_content = content.decode('utf-8', errors='ignore')
        
    return parse_dimacs(text_content)

print("Generators ready.")'''
b.cells.append(nbf.v4.new_code_cell(code_data))

# Cell 4: Solver
# Using triple single quotes. IMPORTANT: Fixed dtype to float32 for cuSPARSE compat
code_solver = r'''# @title 3. The Solver: `TetraDynamicsGPU`
# Implements the Generalized Higher-Order Cluster Dynamics (m=2, 3, 4) + Residual Triangles.

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
        
        # --- 1. Topology Builder: Decompose Clauses into Tetrahedrons & Triangles ---
        # We greedily find tetrahedrons (cliques of 4 vars sharing >=2 clauses).
        print("Decomposing graph topology...")
        tetras, triangles = self._build_topology(clauses_np)
        print(f"Topology: {len(tetras)} Tetrahedrons, {len(triangles)} Residual Triangles.")
        
        # --- 2. Prepare Data for GPU (Tetrahedrons) ---
        # We flatten the tetrahedron list for vectorized ops.
        # tetras structure: list of dicts {'indices': [4], 'signs': [4], 'active': [4 (bool)], 'm': int}
        
        self.num_tetras = len(tetras)
        if self.num_tetras > 0:
            t_indices = np.array([t['indices'] for t in tetras], dtype=np.int32)
            t_signs   = np.array([t['signs'] for t in tetras], dtype=np.int8)
            t_active  = np.array([t['active'] for t in tetras], dtype=bool)
            t_m       = np.array([t['m'] for t in tetras], dtype=np.int8)
            
            self.t_indices = cp.array(t_indices)
            self.t_signs   = cp.array(t_signs)
            self.t_active  = cp.array(t_active) # Mask: True if node is in Active Set A
            self.t_m       = cp.array(t_m)
        else:
            self.t_indices = cp.empty((0, 4), dtype=cp.int32)
            
        # --- 3. Prepare Data for GPU (Residual Triangles) ---
        self.num_tris = len(triangles)
        if self.num_tris > 0:
            r_indices = np.array([t['indices'] for t in triangles], dtype=np.int32)
            r_signs   = np.array([t['signs'] for t in triangles], dtype=np.int8)
            
            self.r_indices = cp.array(r_indices)
            self.r_signs   = cp.array(r_signs)
        else:
            self.r_indices = cp.empty((0, 3), dtype=cp.int32)

        # Initialize Spins
        self.spins = cp.random.choice(cp.array([-1, 1], dtype=cp.int8), size=N + 1)
        # We don't strictly need a dummy node for this logic, but we keep structure consistent
        # if we ever need it. The Ghost nodes are virtual (N+1, N+2).
        
        # Ghost Node Indices
        self.GHOST_PLUS = N + 1
        self.GHOST_MINUS = N + 2
        self.TOTAL_NODES = N + 3
        
        # --- 4. Precompute Probabilities ---
        # We need thresholds for B=0, B=1, B=2 based on U and m.
        # Store as dictionaries or small arrays? 
        # Since we vectorize, we will compute probabilities on the fly or look them up.
        # Constants:
        self.exp_w  = np.exp(-1.0 * omega)
        self.exp_2w = np.exp(-2.0 * omega)
        self.exp_3w = np.exp(-3.0 * omega)
        self.exp_4w = np.exp(-4.0 * omega)

    def _build_topology(self, clauses):
        """
        Decomposes the clause list into Tetrahedrons (m=2,3,4) and Residual Triangles.
        Greedy strategy: Prioritize higher m.
        """
        # 1. Map Edges (pair of literals) to Clauses
        # "Edge" here means two variables sharing a link in the factor graph.
        # The prompt specifies: "two triangles share an edge (two signed literals)".
        # So we key by (lit1, lit2) sorted.
        
        from collections import defaultdict
        
        # Each clause is identified by its index in the original list
        # Store clauses as sets of literals
        clause_sets = []
        for c in clauses:
            clause_sets.append(tuple(sorted(c)))
            
        active_clauses = set(range(len(clauses)))
        
        # Map: Edge (lit_a, lit_b) -> set of clause_indices
        edge_map = defaultdict(set)
        for idx, literals in enumerate(clause_sets):
            # 3 edges per clause
            l = literals
            edge_map[tuple(sorted((l[0], l[1])))].add(idx)
            edge_map[tuple(sorted((l[0], l[2])))].add(idx)
            edge_map[tuple(sorted((l[1], l[2])))].add(idx)
            
        tetras = []
        
        # Search for Tetrahedrons
        # Potential tetrahedrons are formed by pairs of clauses sharing an edge.
        # We iterate edges that have >= 2 clauses.
        
        # To avoid duplicates, we track "consumed" clauses.
        # Heuristic: Check edges with most clauses first? Or just iterate.
        
        # Because N might be large, we need to be efficient. 
        # We'll just iterate all edges with size >= 2.
        
        candidates = [] # (m, clause_indices_tuple, literals_tuple)
        
        checked_pairs = set()

        for edge, c_indices in edge_map.items():
            if len(c_indices) < 2:
                continue
            
            # Check all pairs in this edge list
            c_list = list(c_indices)
            for i in range(len(c_list)):
                for j in range(i + 1, len(c_list)):
                    idx1, idx2 = c_list[i], c_list[j]
                    
                    # Form a candidate 4-set
                    l1 = set(clause_sets[idx1])
                    l2 = set(clause_sets[idx2])
                    union_l = l1 | l2
                    
                    if len(union_l) != 4:
                        # Should be 4 if they share exactly 2 literals (an edge)
                        continue
                        
                    key = tuple(sorted(list(union_l)))
                    if key in checked_pairs:
                        continue
                    checked_pairs.add(key)
                    
                    # Now check how many faces of this 4-set exist in 'clause_sets'
                    # The 4-set has 4 faces (triplets).
                    # We count how many are in our full list.
                    
                    # Generate 4 faces
                    import itertools
                    faces_found = [] # store indices
                    
                    # Map triplet -> original index? Slow to search list.
                    # We can't easily search `clause_sets` if not hashed.
                    # Let's verify presence using a set of all clauses.
                    
                    # Optimization: Only check this if we haven't consumed these.
                    # But we are in the candidate gathering phase.
                    
                    # Count 'm'
                    m_count = 0
                    faces_indices = []
                    
                    # Check the 4 combinations
                    u_list = sorted(list(union_l))
                    possible_faces = list(itertools.combinations(u_list, 3))
                    
                    found_indices = []
                    
                    # We need to find the ID of these faces.
                    # Build a lookup: triplet -> ID
                    # Doing this once at start
                    
                    pass 
        
        # --- optimized topology pass ---
        # Re-build for speed
        
        clause_to_id = {c: i for i, c in enumerate(clause_sets)}
        
        candidates = []
        
        # Iterating edges again
        seen_tetra = set()
        
        for edge, c_indices in edge_map.items():
            if len(c_indices) < 2: continue
            c_list = list(c_indices)
            
            for i in range(len(c_list)):
                for j in range(i+1, len(c_list)):
                    idx1 = c_list[i]
                    idx2 = c_list[j]
                    
                    l1 = set(clause_sets[idx1])
                    l2 = set(clause_sets[idx2])
                    union_l = sorted(list(l1 | l2))
                    
                    if len(union_l) != 4: continue
                    
                    t_key = tuple(union_l)
                    if t_key in seen_tetra: continue
                    seen_tetra.add(t_key)
                    
                    # Check faces
                    # We know idx1 and idx2 are present.
                    # Check the other 2 possible faces.
                    
                    import itertools
                    faces = list(itertools.combinations(union_l, 3))
                    
                    found_faces = []
                    for face in faces:
                        f_tuple = tuple(sorted(face))
                        if f_tuple in clause_to_id:
                            found_faces.append(clause_to_id[f_tuple])
                            
                    m = len(found_faces)
                    if m >= 2:
                        candidates.append({
                            'm': m,
                            'clauses': found_faces,
                            'literals': union_l
                        })
                        
        # Sort candidates by m descending
        candidates.sort(key=lambda x: x['m'], reverse=True)
        
        # Greedy Assignment
        final_tetras = []
        consumed_mask = np.zeros(len(clauses), dtype=bool)
        
        for cand in candidates:
            # Check if any clause is already consumed
            if any(consumed_mask[idx] for idx in cand['clauses']):
                continue
                
            # Consume
            for idx in cand['clauses']:
                consumed_mask[idx] = True
                
            # Build Tetra Object
            # Need to define 'Active' mask.
            # "Active" node is one opposite to an Active Face.
            # Active Face = a clause that exists (is in cand['clauses']).
            
            # Map literals to 0..3 local indices
            lits = cand['literals'] # the 4 literals (signed)
            
            # Store indices (abs(lit)-1) and signs
            indices = [abs(x)-1 for x in lits]
            signs = [int(np.sign(x)) for x in lits]
            
            active_mask = [False] * 4
            
            # Check each vertex v. If face opposite to v is in 'cand['clauses']', v is active.
            # Face opposite to local index i is the triplet excluding i.
            import itertools
            for i in range(4):
                # Form face excluding i
                face_lits = [lits[k] for k in range(4) if k != i]
                face_tuple = tuple(sorted(face_lits))
                
                # Is this face in our consumed list?
                # We need to check if face_tuple corresponds to one of the consumed IDs
                # Using lookup
                if face_tuple in clause_to_id:
                    if clause_to_id[face_tuple] in cand['clauses']:
                        active_mask[i] = True
            
            final_tetras.append({
                'indices': indices,
                'signs': signs,
                'active': active_mask,
                'm': cand['m']
            })
            
        # Collect Residuals
        final_tris = []
        for i in range(len(clauses)):
            if not consumed_mask[i]:
                c = clause_sets[i]
                final_tris.append({
                    'indices': [abs(x)-1 for x in c],
                    'signs': [int(np.sign(x)) for x in c]
                })
                
        return final_tetras, final_tris

    def step(self):
        """
        Swendsen-Wang Step with Generalized Dynamics.
        """
        # --- 1. Tetrahedrons Dynamics ---
        if self.num_tetras > 0:
            # Gather spins (T, 4)
            t_spins = self.spins[self.t_indices]
            # Check sat (T, 4)
            is_sat = (t_spins == self.t_signs)
            # k: num sat (T,)
            k = cp.sum(is_sat, axis=1)
            
            # Determine Energy State U
            # Conditions:
            # U = 0 if (k >= 2) OR (k==1 AND sat_node is Inactive)
            # U = w if (k == 1 AND sat_node is Active)
            # U = m*w if (k == 0)
            
            # Identify if the single satisfied node is Active (for k=1 case)
            # sat_and_active = is_sat & self.t_active
            # is_sat_active_any = cp.any(sat_and_active, axis=1)
            
            # We can compute U directly or implicit probabilities.
            # Let's map to a State Index S:
            # 0: Energy mw (k=0)
            # 1: Energy w  (k=1, Active)
            # 2: Energy 0  (k>=2 or k=1 Inactive)
            
            state = cp.zeros(self.num_tetras, dtype=cp.int8) # Default 0 (k=0)
            
            # Mask k=1
            mask_k1 = (k == 1)
            # Check if the sat node is Active
            # For k=1, exactly one is_sat is True. is_sat & t_active gives True if that one is active.
            mask_k1_active = mask_k1 & cp.any(is_sat & self.t_active, axis=1)
            mask_k1_inactive = mask_k1 & (~mask_k1_active)
            
            mask_k_ge_2 = (k >= 2)
            
            # Assign states
            # State 0 is default
            state[mask_k1_active] = 1
            state[mask_k1_inactive] = 2
            state[mask_k_ge_2] = 2
            
            # Generate Bond B (0, 1, 2)
            # We need vectorized random sampling based on State and m.
            # Probabilities depend on m.
            # We can use a unified random number u.
            
            u = cp.random.random(self.num_tetras, dtype=cp.float32)
            bonds = cp.zeros(self.num_tetras, dtype=cp.int8)
            
            # We must apply rules for m=2,3,4.
            # To vectorize efficiently, we can lookup thresholds based on (m, state).
            # Or handle each m separately. Separating by m is clearer.
            
            for m_val in [2, 3, 4]:
                mask_m = (self.t_m == m_val)
                if not cp.any(mask_m): continue
                
                # Sub-masks
                mask_S0 = mask_m & (state == 0) # U = mw
                mask_S1 = mask_m & (state == 1) # U = w
                mask_S2 = mask_m & (state == 2) # U = 0
                
                # --- State 0 (k=0) ---
                # P(B=0)=1. Always 0.
                
                # --- State 1 (k=1 Active) ---
                # Energy w.
                # P(B=0) = e^{-(m-1)w}
                # P(B=1) = 1 - P(B=0)
                # P(B=2) = 0
                p_b0_s1 = cp.exp(-(m_val - 1) * self.omega)
                
                mask_S1_B1 = mask_S1 & (u >= p_b0_s1)
                bonds[mask_S1_B1] = 1
                
                # --- State 2 (Energy 0) ---
                # P(B=0) = e^{-mw}
                # P(B=1) = e^{-w} - e^{-mw}
                # P(B=2) = 1 - e^{-w}
                
                p_b0_s2 = cp.exp(-m_val * self.omega)
                p_b1_cum_s2 = cp.exp(-1.0 * self.omega) # P(B<=1) = e^{-w}
                
                # B=1 if p_b0 <= u < p_b1_cum
                mask_S2_B1 = mask_S2 & (u >= p_b0_s2) & (u < p_b1_cum_s2)
                bonds[mask_S2_B1] = 1
                
                # B=2 if u >= p_b1_cum
                mask_S2_B2 = mask_S2 & (u >= p_b1_cum_s2)
                bonds[mask_S2_B2] = 2
            
            # --- Witness Selection (Vectorized) ---
            # Generate random priorities for all nodes in tetrahedrons
            priorities = cp.random.random((self.num_tetras, 4), dtype=cp.float32)
            
            # Arrays to store witnesses (Global Indices)
            # We can have up to 2 witnesses per tetra.
            # Initialize with -1 (no witness)
            w1_global = cp.full(self.num_tetras, -1, dtype=cp.int32)
            w2_global = cp.full(self.num_tetras, -1, dtype=cp.int32)
            w1_sign   = cp.zeros(self.num_tetras, dtype=cp.int8)
            w2_sign   = cp.zeros(self.num_tetras, dtype=cp.int8)
            
            # --- Case B=1: Freeze 1 Satisfied Node ---
            mask_active_1 = (bonds >= 1) # B=1 or B=2 both need at least 1 witness
            
            # Priority Logic: "1er sommet satisfait dans l'ordre pi"
            # Mask priorities of unsatisfied nodes to -1
            p_sat = priorities.copy()
            p_sat[~is_sat] = -2.0 # Unsatisfied
            
            # Select Max Priority
            idx_w1 = cp.argmax(p_sat, axis=1) # Local index 0..3
            
            # Store w1 for Active bonds
            w1_global = cp.where(mask_active_1, 
                                 cp.take_along_axis(self.t_indices, idx_w1[:, None], axis=1).flatten(),
                                 w1_global)
            w1_sign = cp.where(mask_active_1,
                               cp.take_along_axis(self.t_signs, idx_w1[:, None], axis=1).flatten(),
                               w1_sign)
            
            # --- Case B=2: Freeze 2nd Node OR Inactive Node ---
            mask_B2 = (bonds == 2)
            
            # Logic:
            # if exists sat node in Inactive (I): Freeze 1 such node (already done in B=1 step? No.)
            # Wait, the rule for B=2 says:
            # "si existe +1 sur I: geler 1 inactif (+1). Sinon: geler 2 sommets (+1)."
            
            # So, for B=2, we must check if we picked an Inactive node as w1?
            # Or does "1er inactif (+1)" imply we prioritize Inactive?
            # "si existe +1 sur I: geler 1 tel sommet (le premier dans pi parmi ceux-là)."
            # This implies a priority filter: Filter for (Sat AND Inactive). If not empty, pick max.
            # Else (if only Active sat), pick top 2 Sat.
            
            # We need to refine w1 selection for B=2 specifically.
            # Let's re-calculate w1, w2 for B=2 rows.
            
            if cp.any(mask_B2):
                # Subset of data for B=2
                # This is tricky to do in-place vectorized without masking.
                # Let's just adjust the priorities for B=2 case before argmax?
                # No, B=1 and B=2 have different selection rules for w1.
                
                # Correction:
                # B=1: Any Sat.
                # B=2: Priority to Inactive Sat.
                
                # Let's compute specific targets for B=2
                
                # Check existence of Sat+Inactive
                has_sat_inactive = cp.any(is_sat & (~self.t_active), axis=1)
                
                # Sub-mask for B=2
                mask_B2_Inactive = mask_B2 & has_sat_inactive
                mask_B2_ActiveOnly = mask_B2 & (~has_sat_inactive)
                
                # For mask_B2_Inactive: Pick max priority among (Sat & Inactive)
                # We can modify p_sat temporary for these rows?
                # Better: construct specific mask.
                
                # 1. Update w1 for B=2 & Inactive
                # mask out Active nodes in priority
                p_sat_inactive = p_sat.copy()
                p_sat_inactive[self.t_active] = -2.0 # Mask active
                
                idx_w1_inactive = cp.argmax(p_sat_inactive, axis=1)
                
                # Apply update
                w1_global = cp.where(mask_B2_Inactive, 
                                     cp.take_along_axis(self.t_indices, idx_w1_inactive[:,None], axis=1).flatten(),
                                     w1_global)
                w1_sign = cp.where(mask_B2_Inactive,
                                   cp.take_along_axis(self.t_signs, idx_w1_inactive[:,None], axis=1).flatten(),
                                   w1_sign)
                                   
                # w2 remains -1 for this case (only 1 witness needed)
                
                # 2. Update w1, w2 for B=2 & Active Only (Need 2 witnesses)
                # w1 is already picked correctly (Max of Sat) because only Active are Sat.
                # We just need w2 (2nd Max of Sat).
                
                # Mask the chosen w1
                p_sat_w2 = p_sat.copy()
                rows = cp.arange(self.num_tetras)
                # We need the local index of the current w1 to mask it.
                # Recover local index?
                # argmax was used on p_sat.
                idx_w1_current = cp.argmax(p_sat, axis=1)
                
                p_sat_w2[rows, idx_w1_current] = -3.0
                
                idx_w2 = cp.argmax(p_sat_w2, axis=1)
                
                # Apply
                w2_global = cp.where(mask_B2_ActiveOnly,
                                     cp.take_along_axis(self.t_indices, idx_w2[:,None], axis=1).flatten(),
                                     w2_global)
                w2_sign = cp.where(mask_B2_ActiveOnly,
                                   cp.take_along_axis(self.t_signs, idx_w2[:,None], axis=1).flatten(),
                                   w2_sign)

            # Store edges for graph
            # Active B >= 1
            mask_has_w1 = (bonds >= 1)
            mask_has_w2 = (w2_global != -1)
            
            src_t1 = w1_global[mask_has_w1]
            tgt_t1 = cp.where(w1_sign[mask_has_w1] > 0, self.GHOST_PLUS, self.GHOST_MINUS)
            
            src_t2 = w2_global[mask_has_w2]
            tgt_t2 = cp.where(w2_sign[mask_has_w2] > 0, self.GHOST_PLUS, self.GHOST_MINUS)
        
        else:
            # Empty placeholders
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
            # P(Freeze) = 1 - e^-w
            p_freeze = 1.0 - self.exp_w
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
            # No bonds? Just flip everything randomly? 
            # Standard SW: If no edges, every node is a cluster.
            # Here we haven't built the full graph of N nodes, only witnessing edges.
            # Implicitly, nodes not in 'all_src' are singletons.
            # They should flip 50/50.
            # The code above relies on 'labels' covering all nodes.
            # If adj is empty, connected_components returns N components (if shape is correct).
            # But coo_matrix shape is (TOTAL, TOTAL), so it should work.
            
            # However, empty COO might be tricky.
            pass
            
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

# Configuration
SOURCE = "SATLIB (uf250)" # @param ["Random", "SATLIB (uf250)", "Custom URL"]
CUSTOM_URL = "" # @param {type:"string"}

# Random Params
N = 2000          # Number of variables (for Random)
alpha = 4.2       # Clause density (for Random)

# Solver Params
steps = 500       # Simulation steps
omega = 3.5       # Interaction strength (Tetra)
beta_base = 4.0   # Inv Temp (Metropolis)
compare_baseline = True # @param {type:"boolean"} 

# Load Data
if SOURCE == "Random":
    print(f"Generating Random 3-SAT: N={N}, M={int(alpha*N)}...")
    clauses, real_N = generate_random_3sat(N, alpha, seed=42)
elif SOURCE == "SATLIB (uf250)":
    # Example hard instance from SATLIB
    url = "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf250-1065.tar.gz"
    print(f"Fetching {url}...")
    clauses, real_N = download_and_parse_instance(url)
else:
    if not CUSTOM_URL:
        print("Error: Please provide a Custom URL.")
        clauses, real_N = np.array([]), 0
    else:
        clauses, real_N = download_and_parse_instance(CUSTOM_URL)

if len(clauses) == 0:
    print("No valid clauses found. Exiting.")
else:
    print(f"Loaded Instance: N={real_N}, M={len(clauses)}")

    # --- Run Tetra Dynamics ---
    print("Initializing TetraDynamicsGPU...")
    tetra_solver = TetraDynamicsGPU(clauses, real_N, omega=omega)

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
    plt.title(rf'3-SAT Optimization: N={real_N}, M={len(clauses)}')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()"""
b.cells.append(nbf.v4.new_code_cell(code_exec))

with open('sw3sat_colab.ipynb', 'w') as f:
    nbf.write(b, f)