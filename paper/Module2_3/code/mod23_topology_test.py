"""
MODULE 2.3 — TOPOLOGY TEST
Is the plateau C set by coordination number k?

Run this on your PC. It's the definitive test for Gap 1.
Takes ~20-40 min depending on hardware.

WHAT TO CHANGE:
  STEPS = 50000   # increase for cleaner fits (try 100000)
  L_2D  = 30      # 2D lattice size (30x30=900 nodes)
  L_3D  = 20      # 3D lattice size (20x20x20=8000 nodes)
  N_SEEDS = 5     # runs per config (increase for error bars)

WHAT TO LOOK FOR:
  C_2D vs C_3D — do they differ?
  If C_2D ≈ C_3D: plateau is NOT topology-dependent
  If C_2D ≠ C_3D: plateau IS topology-dependent
  
  Also check: does C change with lattice size L?
  If C(L=20) ≈ C(L=30) ≈ C(L=40): it's a true fixed point
  If C grows with L: finite-size artifact, need bigger lattice

The key number to report: C_3D - C_2D and its significance.
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import curve_fit
from collections import defaultdict

# ──────────────────────────────────────────────────────────
# PARAMETERS — change these
# ──────────────────────────────────────────────────────────
STEPS    = 50000   # total steps per run
SNAP     = 2000    # record ratio every N steps
N_SEEDS  = 3       # independent runs per config (for error bars)
L_2D     = 30      # 2D lattice side length
L_3D     = 20      # 3D lattice side length

# Fold parameters (keep at baseline)
KAPPA    = 2.0
ALPHA_J  = 0.8
D        = 0.015
P        = 0.25
TAU_RATE = 0.12
ALPHA_R  = 0.4
# ──────────────────────────────────────────────────────────

np.random.seed(42)

def build_lattice(L, dim):
    if dim == 2:
        N = L*L
        rows, cols = [], []
        for x in range(L):
            for y in range(L):
                i = x*L+y
                for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nx,ny = x+dx, y+dy
                    if 0<=nx<L and 0<=ny<L:
                        rows.append(i); cols.append(nx*L+ny)
        edge_nodes = [i for i in range(N)
                      if i//L==0 or i//L==L-1
                      or i%L==0 or i%L==L-1]
        # Interior bulk coordination
        bulk_k = 4  # interior 2D node has 4 neighbors
    else:
        N = L**3
        rows, cols = [], []
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    i = x*L*L+y*L+z
                    for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),
                                      (0,-1,0),(0,0,1),(0,0,-1)]:
                        nx,ny,nz = x+dx,y+dy,z+dz
                        if 0<=nx<L and 0<=ny<L and 0<=nz<L:
                            rows.append(i); cols.append(nx*L*L+ny*L+nz)
        edge_nodes = [i for i in range(N)
                      if any(c==0 or c==L-1
                             for c in [i//(L*L),(i//L)%L,i%L])]
        bulk_k = 6  # interior 3D node has 6 neighbors

    rows = np.array(rows); cols = np.array(cols)
    A = sp.csr_matrix((np.ones(len(rows)),(rows,cols)), shape=(N,N))
    deg = np.array(A.sum(axis=1)).flatten()
    nb = defaultdict(list)
    for r,c in zip(rows,cols): nb[r].append(c)

    return N, rows, cols, A, deg, nb, edge_nodes, bulk_k


def run_one(N, rows, cols, A, deg, nb, edge_nodes, seed=0):
    np.random.seed(seed)
    dt = 0.01/deg.max()
    m = np.ones(N)*5.0 + 0.2*np.random.randn(N)
    m = np.clip(m, 0.1, None)
    tau = np.random.uniform(0, KAPPA*0.2, N)

    ratios = []; steps_r = []

    for step in range(1, STEPS+1):
        diff = m[cols]-m[rows]
        abs_diff = np.abs(diff)
        m_scale = m.max()+1e-10
        weight = np.clip(1.0-ALPHA_J*abs_diff/m_scale, 0.05, 1.0)
        flow = D*diff*weight
        dm_arr = np.bincount(rows, weights=flow, minlength=N)
        m += dt*dm_arr; m = np.clip(m, 0, 100.0)

        grad_node = np.bincount(rows,weights=abs_diff,minlength=N)/(deg+1e-10)
        tau += TAU_RATE*grad_node*dt
        excess = np.maximum(0, tau-KAPPA)
        tau -= ALPHA_R*excess*dt; tau = np.clip(tau, 0, None)
        fired = np.where(excess>0.05)[0]
        for i in fired:
            total = 0.0
            pull = P*min(excess[i]/KAPPA, 1.0)
            for j in nb[i]:
                draw = min(pull*m[j], m[j]*0.35)
                m[j] -= draw; total += draw
            m[i] = min(m[i]+total, 100.0)

        if step % SNAP == 0:
            peak = np.argmax(m)
            m_edge = np.mean(m[edge_nodes])
            ratio = m[peak]/m_edge if m_edge > 0 else 1.0
            ratios.append(ratio); steps_r.append(step)

    ratios = np.array(ratios)
    steps_arr = np.array(steps_r, dtype=float)

    # Fit exponential plateau
    try:
        def exp_dec(t, A, tau_fit, C):
            return A*np.exp(-t/tau_fit)+C
        popt, pcov = curve_fit(
            exp_dec, steps_arr, ratios,
            p0=[ratios[0]-ratios[-1], STEPS/2, ratios[-1]],
            maxfev=5000)
        C = float(popt[2])
        C_err = float(np.sqrt(pcov[2,2])) if pcov[2,2] > 0 else 0.001
        return C, C_err, ratios, steps_arr
    except:
        return float(ratios[-1]), 0.005, ratios, steps_arr


# ── RUN CONFIGS ────────────────────────────────────────────
configs = [
    (2, L_2D, f'2D L={L_2D} (k=4 bulk)'),
    (3, L_3D, f'3D L={L_3D} (k=6 bulk)'),
]

# Also size sweep for 3D
size_sweep = [15, 20, 25, 30]

print("="*62)
print("MODULE 2.3 TOPOLOGY TEST")
print(f"Steps={STEPS}, N_seeds={N_SEEDS}, Snap={SNAP}")
print("="*62)

import time

# ── 2D vs 3D ───────────────────────────────────────────────
print("\n[1] 2D vs 3D comparison")
print(f"{'Config':>25} {'k_bulk':>7} {'C mean':>10} "
      f"{'C std':>8} {'C-1':>8}")
print("-"*62)

dim_results = {}
for dim, L, label in configs:
    t0 = time.time()
    N,rows,cols,A,deg,nb,edge_nodes,bulk_k = build_lattice(L,dim)
    Cs = []
    for seed in range(N_SEEDS):
        C, C_err, _, _ = run_one(N,rows,cols,A,deg,nb,edge_nodes,seed=seed)
        Cs.append(C)
        print(f"  {label} seed={seed}: C={C:.5f}", flush=True)
    C_m = np.mean(Cs); C_s = np.std(Cs)
    print(f"  {'→ MEAN':>23}  {bulk_k:>7}  {C_m:>10.5f}  "
          f"{C_s:>8.5f}  {C_m-1:>8.5f}")
    print(f"  Time: {time.time()-t0:.0f}s")
    dim_results[dim] = (C_m, C_s, bulk_k)

# Key comparison
C_2D, s_2D, k_2D = dim_results[2]
C_3D, s_3D, k_3D = dim_results[3]
delta = C_3D - C_2D
sigma = np.sqrt(s_2D**2 + s_3D**2)
print(f"\n  C_3D - C_2D = {delta:+.5f}  (σ = {sigma:.5f})")
if abs(delta) > 2*sigma:
    print(f"  → SIGNIFICANT: plateau is topology-dependent")
    print(f"    Higher k → {'higher' if delta>0 else 'lower'} plateau")
else:
    print(f"  → NOT SIGNIFICANT: plateau is parameter-universal")
    print(f"    C ≈ {(C_2D+C_3D)/2:.4f} independent of dimension")

# ── 3D SIZE SWEEP ─────────────────────────────────────────
print(f"\n[2] 3D size sweep (finite-size check)")
print(f"{'L':>5} {'N':>8} {'C':>10} {'C-1':>10}")
print("-"*38)

size_results = []
for L in size_sweep:
    N,rows,cols,A,deg,nb,edge_nodes,_ = build_lattice(L,3)
    C,C_err,_,_ = run_one(N,rows,cols,A,deg,nb,edge_nodes,seed=0)
    print(f"  {L:>4}  {N:>8}  {C:>10.5f}  {C-1:>10.5f}")
    size_results.append((L,N,C))

# Check finite-size scaling
Ls = np.array([r[0] for r in size_results])
Cs = np.array([r[2] for r in size_results])
if len(Ls) >= 3:
    sl,_ = np.polyfit(1/Ls, Cs, 1)
    C_inf = Cs[-1] - sl/Ls[-1]  # rough extrapolation
    print(f"\n  Extrapolated C(L→∞) ≈ {C_inf:.5f}")
    print(f"  Finite-size slope dC/d(1/L) = {sl:.4f}")
    if abs(sl) < 0.05:
        print(f"  → WEAK finite-size effect: C is converged")
    else:
        print(f"  → SIGNIFICANT finite-size effect: run larger L")

print("\n" + "="*62)
print("SUMMARY FOR PAPER")
print("="*62)
print(f"""
  Topology comparison (2D k=4 vs 3D k=6):
    C_2D = {C_2D:.5f} ± {s_2D:.5f}
    C_3D = {C_3D:.5f} ± {s_3D:.5f}
    Δ = {delta:+.5f}  ({abs(delta/sigma):.1f}σ)

  Size convergence (3D):
    L=15: C={size_results[0][2]:.5f}
    L=30: C={size_results[-1][2]:.5f}
    
  Bottom line:
    {"Plateau is topology-dependent" if abs(delta)>2*sigma
     else "Plateau is universal within measurement precision"}
    C ≈ {C_3D:.3f} on 3D cubic fold lattice
""")
print("Done. Send results to Green for paper update.")
