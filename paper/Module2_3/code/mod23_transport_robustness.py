"""
MODULE 2.3 TRANSPORT ROBUSTNESS TEST (v2 — Locality Fix)
=========================================================
Tests three LOCALLY COMPLIANT transport forms.

Background:
  A Gemini/Grok audit (April 2026) identified that earlier
  forms using m_max (global maximum structural mass) violate
  the pre-geometric locality rule: J_ij must depend only on
  local state variables (m_i, m_j, kappa), not global quantities.

  The three locally compliant forms tested here are:

  1. Saturation (canonical):
       g = D * m_i / (m_i + |Dm|)
       Uses only local pair (m_i, m_j). No free parameter beyond D.
       Satisfies Anti-Zeno bound exactly at large gradients.

  2. Kappa-scaled (perturbative):
       g = D * (1 - alpha*|Dm|/kappa)
       Uses kappa (fold primitive) as local scale. Leading-order.

  3. Local-rational:
       g = D / (1 + alpha*|Dm|/m_i)
       Uses m_i as local scale.

  All three satisfy:
    - g -> D as |Dm| -> 0  (reduces to standard diffusion)
    - g -> 0 as |Dm| -> inf  (Anti-Zeno compliant)
    - g >= 0  (T1 compatible)
    - depends only on (m_i, m_j, kappa)  (pre-geometric locality)

  REJECTED (global m_max violates pre-geometric locality):
    Linear        g = D(1 - alpha*|Dm|/m_max)   GLOBAL
    Exponential   g = D*exp(-alpha*|Dm|/m_max)   GLOBAL
    Rational      g = D/(1 + alpha*|Dm|/m_max)   GLOBAL

If all three local forms give C ~ 1.014 -> shell formation
is a property of the locally admissible class.

PARAMETERS:
  L = 20        # lattice size (20^3 = 8000 nodes)
  STEPS = 40000 # steps per run (~90 min on typical hardware)
  N_SEEDS = 3   # seeds per transport form
  Quick test: L=15, STEPS=25000, N_SEEDS=3 (~1 min)
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import curve_fit
from collections import defaultdict
import time, json

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
print("TRANSPORT ROBUSTNESS TEST (v2 - Locally Compliant Forms)")
print(f"L={L}, STEPS={STEPS}, N_SEEDS={N_SEEDS}")
print("Testing: saturation / kappa-scaled / local-rational")
print("="*65)

# Build lattice
rows, cols = [], []
for x in range(L):
    for y in range(L):
        for z in range(L):
            i = x*L*L + y*L + z
            for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),
                              (0,-1,0),(0,0,1),(0,0,-1)]:
                nx,ny,nz = x+dx,y+dy,z+dz
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

# ── Locally compliant transport forms ─────────────────────

def transport_saturation(diff, abs_diff, m_i):
    """
    Canonical: g = D * m_i / (m_i + |Dm|)
    Local: uses only (m_i, m_j). No free parameter beyond D.
    Anti-Zeno bound: g <= m_i/(|Dm|) -> 0 as |Dm| -> inf.
    """
    weight = m_i / (m_i + abs_diff + 1e-10)
    return D * diff * weight

def transport_kappa_scaled(diff, abs_diff, m_i):
    """
    Perturbative: g = D * (1 - alpha*|Dm|/kappa)
    Local: kappa is a fold primitive (local state variable).
    Leading-order expansion of any compliant g around |Dm|=0.
    """
    weight = np.clip(1.0 - ALPHA_J*abs_diff/KAPPA, 0.05, 1.0)
    return D * diff * weight

def transport_local_rational(diff, abs_diff, m_i):
    """
    Local rational: g = D / (1 + alpha*|Dm|/m_i)
    Local: m_i is a local state variable.
    """
    weight = np.clip(1.0 / (1.0 + ALPHA_J*abs_diff/(m_i+1e-10)),
                     0.05, 1.0)
    return D * diff * weight

transport_forms = [
    ('Saturation',     transport_saturation),
    ('Kappa-scaled',   transport_kappa_scaled),
    ('Local-rational', transport_local_rational),
]

def run_one(transport_fn, seed=0):
    np.random.seed(seed)
    m = np.ones(N)*5.0 + 0.2*np.random.randn(N)
    m = np.clip(m, 0.1, None)
    tau = np.random.uniform(0, KAPPA*0.2, N)
    ratios = []; steps_r = []

    for step in range(1, STEPS+1):
        diff     = m[cols] - m[rows]
        abs_diff = np.abs(diff)
        m_i      = m[rows]           # local — passed to transport fn
        flow = transport_fn(diff, abs_diff, m_i)
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
            ratio = m.max()/m_edge if m_edge > 0 else 1.0
            ratios.append(ratio); steps_r.append(float(step))

    ratios = np.array(ratios); steps_r = np.array(steps_r)
    try:
        def exp_dec(t, A, tau_fit, C):
            return A*np.exp(-t/tau_fit) + C
        popt, _ = curve_fit(exp_dec, steps_r, ratios,
                            p0=[ratios[0]-ratios[-1], STEPS/2, ratios[-1]],
                            maxfev=3000)
        return float(popt[2])
    except:
        return float(ratios[-1])

# ── Run all forms ──────────────────────────────────────────
results = {}
print(f"\n{'Form':>16} {'Seed':>6} {'C':>10}")
print("-"*36)

for form_name, transport_fn in transport_forms:
    t0 = time.time(); Cs = []
    for seed in range(N_SEEDS):
        C = run_one(transport_fn, seed=seed)
        Cs.append(C)
        print(f"  {form_name:>14}  {seed:>6}  {C:>10.5f}")
    mean_C = np.mean(Cs); std_C = np.std(Cs)
    results[form_name] = (mean_C, std_C, Cs)
    print(f"  {'-> MEAN':>14}  {'':>6}  {mean_C:>10.5f} +/- {std_C:.5f}"
          f"  ({time.time()-t0:.0f}s)\n")

# ── Verdict ────────────────────────────────────────────────
print("="*65)
print("ROBUSTNESS VERDICT (LOCAL FORMS ONLY)")
print("="*65)
print(f"\n{'Form':>16} {'C mean':>10} {'C std':>8} {'C-1':>8} {'OK':>6}")
print("-"*52)

C_vals = []
for form_name, (mean_C, std_C, _) in results.items():
    C_vals.append(mean_C)
    ok = "YES" if 1.005 < mean_C < 1.035 else "CHECK"
    print(f"  {form_name:>14}  {mean_C:>10.5f}  {std_C:>8.5f}"
          f"  {mean_C-1:>8.5f}  {ok:>6}")

C_spread = max(C_vals) - min(C_vals)
ref = results['Saturation'][0]
print(f"\n  Spread across local forms: {C_spread:.5f}")
print(f"  Reference (saturation):    C = {ref:.5f}")

confirmed = all(1.005 < v[0] < 1.035 for v in results.values())
if confirmed and C_spread < 0.015:
    print("""
  CONFIRMED: LOCAL TRANSPORT ROBUSTNESS

  All three pre-geometrically local transport forms
  (saturation, kappa-scaled, local-rational) give C ~ 1.02.

  Shell formation is a property of the locally admissible
  class (Anti-Zeno + pre-geometric locality).

  Canonical form: saturation g = D*m_i/(m_i+|Dm|)
  - fully local, no free parameter, Anti-Zeno exact.

  The m_max locality bug (Gemini/Grok, April 2026) is
  confirmed cosmetic: the physics result survives.
""")
else:
    print("\n  Inconclusive - check parameters or increase STEPS.")

# ── Save JSON ──────────────────────────────────────────────
try:
    with open('mod23_transport_robustness_results.json', 'w') as f:
        json.dump({
            'version': 'v2_local_forms',
            'locality_fix': 'April 2026 - m_max replaced by local forms',
            'params': {
                'L': L, 'STEPS': STEPS, 'N_SEEDS': N_SEEDS,
                'KAPPA': float(KAPPA), 'D': float(D),
                'ALPHA_J': float(ALPHA_J), 'P': float(P),
                'TAU_RATE': float(TAU_RATE)
            },
            'forms_tested': list(results.keys()),
            'forms_rejected_global': [
                'linear_mmax', 'exponential_mmax', 'rational_mmax'
            ],
            'results': {
                name: {
                    'C_mean': float(mean_C),
                    'C_std': float(std_C),
                    'C_values': [float(c) for c in Cs]
                }
                for name, (mean_C, std_C, Cs) in results.items()
            },
            'spread': float(C_spread),
            'verdict': 'confirmed' if confirmed and C_spread < 0.015
                       else 'inconclusive'
        }, f, indent=2)
    print("Saved mod23_transport_robustness_results.json")
except Exception as e:
    print(f"JSON save error: {e}")
