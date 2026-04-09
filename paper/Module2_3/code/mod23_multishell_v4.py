"""
MODULE 2.3 MULTI-SHELL v4
Key insight from v3: at steps 3000-6000, ALL shell criteria passed.
The collapse cascade at step 9000 destroyed the structure.

Strategy: gentle enough that shells form stably for the full run.
B_interior will drift below B_edge naturally over time (as in
the single-shell long run where plateau was 1.021).

Changes from v3:
  INNER_BOOST = 1.20   (was 1.35 — too strong, caused runaway)
  MAX_PULL    = 0.08   (was 0.20 — tighter cap on collapse pull)
  STEPS       = 80000  (longer to catch natural B depletion)
"""
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import time, json

# ──────────────────────────────────────────────────────────
L            = 40
STEPS        = 80000
SNAP         = 4000

KAPPA        = 2.0
ALPHA_J      = 0.8
D            = 0.015
P            = 0.25
TAU_RATE     = 0.12
ALPHA_R      = 0.4

OUTER_RADIUS = 8
OUTER_BOOST  = 1.05
INNER_RADIUS = 2
INNER_BOOST  = 1.20     # gentle — avoids runaway
N_INNER      = 3
MAX_PULL     = 0.08     # tight cap — prevents cascade
# ──────────────────────────────────────────────────────────

np.random.seed(42)
N=L**3; CX=CY=CZ=L//2

print("="*65)
print("MODULE 2.3 MULTI-SHELL v4 — GENTLE, STABLE")
print(f"L={L}, STEPS={STEPS}")
print(f"INNER_BOOST={INNER_BOOST}, MAX_PULL={MAX_PULL}")
print(f"Goal: stable shells + natural B depletion over time")
print("="*65)

print("\nBuilding lattice...", flush=True)
t0=time.time()
rows,cols=[],[]
for x in range(L):
    for y in range(L):
        for z in range(L):
            i=x*L*L+y*L+z
            for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),
                              (0,-1,0),(0,0,1),(0,0,-1)]:
                nx,ny,nz=x+dx,y+dy,z+dz
                if 0<=nx<L and 0<=ny<L and 0<=nz<L:
                    rows.append(i); cols.append(nx*L*L+ny*L+nz)
rows=np.array(rows); cols=np.array(cols)
A_mat=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(N,N))
deg=np.array(A_mat.sum(axis=1)).flatten()
nb=defaultdict(list)
for r,c in zip(rows,cols): nb[r].append(c)
coords=np.array([[i//(L*L),(i//L)%L,i%L] for i in range(N)],dtype=float)

cx_arr=np.array([CX,CY,CZ],dtype=float)
outer_dists=np.sqrt(np.sum((coords-cx_arr)**2,axis=1))
edge_nodes=np.where(outer_dists>=L//2-2)[0]

offset=OUTER_RADIUS//2
inner_centers=[]
for dxyz in [(offset,0,0),(-offset,0,0),(0,offset,0)][:N_INNER]:
    inner_centers.append(tuple(c+d for c,d in zip((CX,CY,CZ),dxyz)))

min_inner_dist=np.full(N,np.inf)
for icx,icy,icz in inner_centers:
    d=np.sqrt(np.sum((coords-np.array([icx,icy,icz],dtype=float))**2,axis=1))
    min_inner_dist=np.minimum(min_inner_dist,d)
void_mask=(outer_dists<=OUTER_RADIUS*0.75)&(min_inner_dist>INNER_RADIUS*2.5)
void_nodes=np.where(void_mask)[0]
print(f"  Void nodes: {len(void_nodes)}, Edge nodes: {len(edge_nodes)}")
print(f"  Done in {time.time()-t0:.1f}s")

# Initial field
m=np.ones(N)*5.0+0.2*np.random.randn(N); m=np.clip(m,0.1,None)
for x in range(L):
    for y in range(L):
        for z in range(L):
            if (x-CX)**2+(y-CY)**2+(z-CZ)**2<=OUTER_RADIUS**2:
                m[x*L*L+y*L+z]*=OUTER_BOOST
for icx,icy,icz in inner_centers:
    for x in range(max(0,icx-INNER_RADIUS),min(L,icx+INNER_RADIUS+1)):
        for y in range(max(0,icy-INNER_RADIUS),min(L,icy+INNER_RADIUS+1)):
            for z in range(max(0,icz-INNER_RADIUS),min(L,icz+INNER_RADIUS+1)):
                if (x-icx)**2+(y-icy)**2+(z-icz)**2<=INNER_RADIUS**2:
                    m[x*L*L+y*L+z]*=INNER_BOOST
tau=np.random.uniform(0,KAPPA*0.2,N)
dt=0.01/deg.max()

def find_peak_and_ratio(m, cx, cy, cz, search_r, ref_nodes):
    """Find peak near (cx,cy,cz) and ratio to reference."""
    c=np.array([cx,cy,cz],dtype=float)
    dists=np.sqrt(np.sum((coords-c)**2,axis=1))
    near=np.where(dists<=search_r)[0]
    if len(near)==0: return 1.0, int(cx)*L*L+int(cy)*L+int(cz)
    peak_idx=near[np.argmax(m[near])]
    ref_mean=float(np.mean(m[ref_nodes]))
    ratio=m[peak_idx]/ref_mean if ref_mean>0 else 1.0
    return float(ratio), int(peak_idx)

def find_boundary(m, cx, cy, cz, max_r):
    c=np.array([cx,cy,cz],dtype=float)
    dists=np.sqrt(np.sum((coords-c)**2,axis=1))
    shells=defaultdict(list)
    for i in range(N):
        r=int(round(dists[i]))
        if r<=max_r: shells[r].append(m[i])
    rs=sorted(shells.keys())
    if len(rs)<4: return None
    means=[np.mean(shells[r]) for r in rs]
    grads=[means[i+1]-means[i] for i in range(len(means)-1)]
    for i in range(len(grads)-1):
        if grads[i]<-0.05 and grads[i+1]>-0.01:
            return rs[i+1]
    return None

print(f"\nRunning {STEPS} steps...", flush=True)
snapshots={}; t0=time.time()

for step in range(1,STEPS+1):
    diff=m[cols]-m[rows]
    abs_diff=np.abs(diff)
    m_scale=m.max()+1e-10
    weight=np.clip(1.0-ALPHA_J*abs_diff/m_scale,0.05,1.0)
    flow=D*diff*weight
    dm_arr=np.bincount(rows,weights=flow,minlength=N)
    m+=dt*dm_arr; m=np.clip(m,0,50.0)   # lower cap = gentler

    grad_node=np.bincount(rows,weights=abs_diff,minlength=N)/(deg+1e-10)
    tau+=TAU_RATE*grad_node*dt
    excess=np.maximum(0,tau-KAPPA)
    tau-=ALPHA_R*excess*dt; tau=np.clip(tau,0,None)
    fired=np.where(excess>0.05)[0]
    for i in fired:
        total=0.0
        pull=min(P*min(excess[i]/KAPPA,1.0), MAX_PULL)
        for j in nb[i]:
            draw=min(pull*m[j],m[j]*0.25)  # also cap neighbor drain to 25%
            m[j]-=draw; total+=draw
        m[i]=min(m[i]+total,50.0)

    if step%SNAP==0:
        elapsed=time.time()-t0
        eta=elapsed/(step/STEPS)-elapsed

        # Outer: find peak near center
        outer_ratio,_=find_peak_and_ratio(m,CX,CY,CZ,
                                           OUTER_RADIUS//2, edge_nodes)
        outer_br=find_boundary(m,CX,CY,CZ,L//3)

        # Inner: find peak near each inner center
        inner_ratios=[]
        for icx,icy,icz in inner_centers:
            c_arr=np.array([icx,icy,icz],dtype=float)
            d_arr=np.sqrt(np.sum((coords-c_arr)**2,axis=1))
            ring=np.where((d_arr>=INNER_RADIUS*1.5)&
                          (d_arr<=INNER_RADIUS*3.5))[0]
            ratio,_=find_peak_and_ratio(m,icx,icy,icz,
                                         INNER_RADIUS+1, ring)
            inner_ratios.append(ratio)

        B_interior=float(np.mean(m[void_nodes])) if len(void_nodes)>0 else 0
        B_edge_val =float(np.mean(m[edge_nodes]))

        print(f"\n  Step {step:>6}  ({elapsed:.0f}s, ~{eta:.0f}s left)")
        print(f"    Outer:      ratio={outer_ratio:.4f}  bdy={outer_br}")
        print(f"    Inner:      {[f'{r:.4f}' for r in inner_ratios]}")
        print(f"    B_interior={B_interior:.4f}  B_edge={B_edge_val:.4f}  "
              f"ratio={B_interior/B_edge_val:.4f}" if B_edge_val>0 else "")
        print(f"    m_max={m.max():.2f}  m_std={m.std():.4f}", flush=True)

        snapshots[step]={
            'outer_ratio': float(outer_ratio),
            'outer_br': int(outer_br) if outer_br is not None else -1,
            'inner_ratios': [float(r) for r in inner_ratios],
            'B_interior': B_interior,
            'B_edge': float(B_edge_val),
            'm_std': float(m.std()),
            'm_max': float(m.max())
        }

# ── ANALYSIS ───────────────────────────────────────────────
print("\n"+"="*65)
print("MULTI-SHELL HIERARCHY ANALYSIS")
print("="*65)
final=snapshots[max(snapshots.keys())]
snap_keys=sorted(snapshots.keys())

outer_brs=[snapshots[k]['outer_br'] for k in snap_keys
           if snapshots[k]['outer_br']>0]
B_ratios=[snapshots[k]['B_interior']/snapshots[k]['B_edge']
          for k in snap_keys if snapshots[k]['B_edge']>0]

C1 = final['outer_ratio']>1.05
C2 = all(r>1.05 for r in final['inner_ratios']) if final['inner_ratios'] else False
C3 = (np.mean(final['inner_ratios'])>final['outer_ratio']
      if final['inner_ratios'] else False)
C4 = final['B_interior']<final['B_edge']*0.98 if final['B_edge']>0 else False
C5 = (len(B_ratios)>=4 and
      np.mean(B_ratios[-2:])<np.mean(B_ratios[:2])-0.003)

print(f"\n  C1 Outer shell stable (>1.05):    {C1}")
print(f"     ratio={final['outer_ratio']:.4f}, "
      f"br stable: {len(outer_brs)>=3}")
print(f"\n  C2 Inner shells elevated (>1.05): {C2}")
print(f"     {[f'{r:.4f}' for r in final['inner_ratios']]}")
print(f"\n  C3 Inner > outer concentration:   {C3}")
print(f"     inner={np.mean(final['inner_ratios']):.4f}  "
      f"outer={final['outer_ratio']:.4f}")
print(f"\n  C4 B_interior < B_edge now:       {C4}")
print(f"     ratio={final['B_interior']/final['B_edge']:.4f}"
      if final['B_edge']>0 else "")
print(f"\n  C5 B ratio declining over time:   {C5}")
if B_ratios:
    print(f"     early={np.mean(B_ratios[:2]):.4f}  "
          f"late={np.mean(B_ratios[-2:]):.4f}  "
          f"trend={'↓' if C5 else '→'}")

all_pass = C1 and C2 and C3 and (C4 or C5)

if all_pass:
    print(f"""
  ✓✓ MULTI-SHELL HIERARCHY CONFIRMED
  Outer shell stable, inner shells more concentrated,
  B_interior depleting relative to B_edge.
  Module 3 nested B argument dynamically confirmed.
  Gap 2 is closed.
""")
elif C1 and C2 and C3:
    print(f"\n  ✓ Shells nested correctly. B depletion {'confirmed' if C4 else 'trend present' if C5 else 'not yet — run longer'}.")
elif C1 and C2:
    print(f"\n  ~ Shells present but inner not more concentrated than outer.")
elif C1:
    print(f"\n  ~ Outer shell only. Increase INNER_BOOST to 1.25.")
else:
    print(f"\n  ✗ No stable shell.")

with open('mod23_multishell_results.json','w') as f:
    json.dump({
        'params':{'L':L,'STEPS':STEPS,'INNER_BOOST':INNER_BOOST,
                  'MAX_PULL':MAX_PULL},
        'criteria':{'C1':C1,'C2':C2,'C3':C3,'C4':C4,'C5':C5,
                    'all_pass':all_pass},
        'final':final,
        'outer_ratios':[float(snapshots[k]['outer_ratio'])
                        for k in snap_keys],
        'inner_means':[float(np.mean(snapshots[k]['inner_ratios']))
                       for k in snap_keys],
        'B_ratios':[float(x) for x in B_ratios],
        'verdict':('confirmed' if all_pass else
                   'shells_nested' if (C1 and C2 and C3) else
                   'partial' if (C1 and C2) else
                   'outer_only' if C1 else 'failed')
    }, f, indent=2)
print("Saved mod23_multishell_results.json — send to Green.")
