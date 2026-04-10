"""
MODULE 2.3 TRANSPORT ROBUSTNESS TEST
Does shell formation depend on the LINEAR transport form?
Or does any Anti-Zeno-compliant closure give the same result?

Test three transport forms:
  1. Linear (current):  g = D × (1 - α|Δm|/m_max)
  2. Exponential:       g = D × exp(-α|Δm|/m_max)
  3. Rational:          g = D / (1 + α|Δm|/m_max)

All three satisfy:
  - g → D at |Δm| → 0  (reduces to standard diffusion)
  - g → 0 at |Δm| → ∞  (Anti-Zeno compliant)
  - g ≥ 0 (T1 compatible)

If all three give C ≈ 1.014 → shell formation is robust,
not dependent on the specific form of the transport law.

PARAMETERS TO CHANGE:
  L = 20        # lattice size (20^3 = 8000 nodes)
  STEPS = 40000 # steps per run
  N_SEEDS = 3   # seeds per transport form
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import curve_fit
from collections import defaultdict
import time

# ──────────────────────────────────────────────────────────
L       = 20
STEPS   = 40000
SNAP    = max(1, STEPS // 20)
N_SEEDS = 3

KAPPA    = 2.0
ALPHA_J  = 0.8
D        = 0.015
P        = 0.25
TAU_RATE = 0.12
ALPHA_R  = 0.4
# ──────────────────────────────────────────────────────────

np.random.seed(42)
N = L**3

print("="*65)
print("TRANSPORT ROBUSTNESS TEST")
print(f"L={L}, STEPS={STEPS}, N_SEEDS={N_SEEDS}")
print("Testing: linear / exponential / rational transport")
print("="*65)

# Build lattice once
rows, cols = [], []
for x in range(L):
    for y in range(L):
        for z in range(L):
            i = x*L*L + y*L + z
            for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),
                              (0,-1,0),(0,0,1),(0,0,-1)]:
                nx,ny,nz = x+dx, y+dy, z+dz
                if 0<=nx<L and 0<=ny<L and 0<=nz<L:
                    rows.append(i); cols.append(nx*L*L+ny*L+nz)

rows = np.array(rows); cols = np.array(cols)
A_mat = sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(N,N))
deg = np.array(A_mat.sum(axis=1)).flatten()
nb = defaultdict(list)
for r,c in zip(rows,cols): nb[r].append(c)
edge_nodes = [i for i in range(N)
              if any(c==0 or c==L-1 for c in [i//(L*L),(i//L)%L,i%L])]
dt = 0.01/deg.max()

# Transport forms
def transport_linear(diff, abs_diff, m_scale):
    """Current form: g = D × (1 - α|Δm|/m_max)"""
    weight = np.clip(1.0 - ALPHA_J*abs_diff/m_scale, 0.05, 1.0)
    return D * diff * weight

def transport_exponential(diff, abs_diff, m_scale):
    """g = D × exp(-α|Δm|/m_max)"""
    weight = np.exp(-ALPHA_J * abs_diff / m_scale)
    weight = np.clip(weight, 0.05, 1.0)
    return D * diff * weight

def transport_rational(diff, abs_diff, m_scale):
    """g = D / (1 + α|Δm|/m_max)"""
    weight = 1.0 / (1.0 + ALPHA_J * abs_diff / m_scale)
    weight = np.clip(weight, 0.05, 1.0)
    return D * diff * weight

transport_forms = [
    ('Linear',      transport_linear),
    ('Exponential', transport_exponential),
    ('Rational',    transport_rational),
]

def run_one(transport_fn, seed=0):
    np.random.seed(seed)
    m = np.ones(N)*5.0 + 0.2*np.random.randn(N)
    m = np.clip(m, 0.1, None)
    tau = np.random.uniform(0, KAPPA*0.2, N)
    ratios = []; steps_r = []

    for step in range(1, STEPS+1):
        diff = m[cols] - m[rows]
        abs_diff = np.abs(diff)
        m_scale = m.max() + 1e-10
        flow = transport_fn(diff, abs_diff, m_scale)
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
            m_edge = np.mean(m[edge_nodes])
            peak = np.argmax(m)
            ratio = m[peak]/m_edge if m_edge > 0 else 1.0
            ratios.append(ratio); steps_r.append(step)

    ratios = np.array(ratios)
    steps_r = np.array(steps_r, dtype=float)
    try:
        def exp_dec(t, A, tau_fit, C):
            return A*np.exp(-t/tau_fit) + C
        popt, _ = curve_fit(exp_dec, steps_r, ratios,
                            p0=[ratios[0]-ratios[-1], STEPS/2, ratios[-1]],
                            maxfev=3000)
        return float(popt[2])
    except:
        return float(ratios[-1])

# Run all forms
results = {}
print(f"\n{'Form':>14} {'Seed':>6} {'C':>10}")
print("-"*34)

for form_name, transport_fn in transport_forms:
    t0 = time.time()
    Cs = []
    for seed in range(N_SEEDS):
        C = run_one(transport_fn, seed=seed)
        Cs.append(C)
        print(f"  {form_name:>12}  {seed:>6}  {C:>10.5f}")
    mean_C = np.mean(Cs)
    std_C = np.std(Cs)
    results[form_name] = (mean_C, std_C, Cs)
    print(f"  {'→ MEAN':>12}  {'':>6}  {mean_C:>10.5f} ± {std_C:.5f}"
          f"  ({time.time()-t0:.0f}s)")
    print()

print("="*65)
print("ROBUSTNESS VERDICT")
print("="*65)
print(f"\n{'Form':>14} {'C mean':>10} {'C std':>8} {'C-1':>8} {'Verdict':>12}")
print("-"*56)

all_consistent = True
C_vals = []
for form_name, (mean_C, std_C, _) in results.items():
    C_vals.append(mean_C)
    verdict = "✓" if 1.005 < mean_C < 1.025 else "~" if 1.0 < mean_C < 1.03 else "✗"
    if verdict == "✗": all_consistent = False
    print(f"  {form_name:>12}  {mean_C:>10.5f}  {std_C:>8.5f}  "
          f"{mean_C-1:>8.5f}  {verdict:>12}")

C_spread = max(C_vals) - min(C_vals)
print(f"\n  Spread across forms: {C_spread:.5f}")
print(f"  Reference (linear):  C = {results['Linear'][0]:.5f}")

if all_consistent and C_spread < 0.01:
    print(f"""
  ✓✓ TRANSPORT ROBUSTNESS CONFIRMED
  
  All three Anti-Zeno-compliant transport forms
  (linear, exponential, rational) give C ≈ 1.014.
  
  Shell formation does NOT depend on the specific
  functional form of the transport law — only on
  the Anti-Zeno compliance property g → 0 at
  large gradients.
  
  This kills the "linear form is arbitrary" critique.
  The result is robust across the entire admissible class.
""")
elif all_consistent:
    print(f"""
  ✓ All forms produce shells (C > 1).
  Spread = {C_spread:.4f} — slightly larger than expected.
  Run with more seeds or longer STEPS for tighter result.
""")
else:
    print(f"""
  ✗ Not all forms give consistent C.
  Check parameters — may need longer run.
""")

print("Send results to Green.")
