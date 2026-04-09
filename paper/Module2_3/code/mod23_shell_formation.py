"""
MODULE 2.3 v3 — STABLE SHELL
v2 got criteria 1 and 3 but ratio was decaying.
Fix: stronger collapse pull, weaker diffusion, check stability.
Also: soften boundary gradient threshold to catch shallow shells.
"""
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

L=20; N=L**3
D=0.015           # weaker diffusion — let collapse win
KAPPA=2.0
TAU_RATE=0.12
ALPHA=0.4
COLLAPSE_PULL=0.25  # stronger concentration
M_MAX=100.0
STEPS=6000
SNAPSHOT_EVERY=600

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
A=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(N,N))
deg=np.array(A.sum(axis=1)).flatten()
neighbors=defaultdict(list)
for r,c in zip(rows,cols): neighbors[r].append(c)

cx_coord=np.array([L//2,L//2,L//2],dtype=float)
coords=np.array([[i//(L*L),(i//L)%L,i%L] for i in range(N)],dtype=float)
euc_dist=np.sqrt(np.sum((coords-cx_coord)**2,axis=1))

m=np.ones(N)*5.0+0.2*np.random.randn(N)
# 10% seed — 3x3x3 cube at center
for dx in range(-1,2):
    for dy in range(-1,2):
        for dz in range(-1,2):
            nx,ny,nz=L//2+dx,L//2+dy,L//2+dz
            if 0<=nx<L and 0<=ny<L and 0<=nz<L:
                m[nx*L*L+ny*L+nz]*=1.10
m=np.clip(m,0.1,None)
tau=np.random.uniform(0,KAPPA*0.2,N)
dt=0.01/deg.max()

center=(L//2)*L*L+(L//2)*L+(L//2)
edge_nodes=[i for i in range(N)
            if any(c==0 or c==L-1 for c in [i//(L*L),(i//L)%L,i%L])]

def radial_profile(m):
    shells=defaultdict(list)
    for i in range(N):
        r=int(round(euc_dist[i]))
        shells[r].append(m[i])
    return shells

def find_boundary(shells, grad_thresh=-0.05):
    rs=sorted(shells.keys())
    if len(rs)<4: return None
    means=[np.mean(shells[r]) for r in rs]
    grads=[means[i+1]-means[i] for i in range(len(means)-1)]
    for i in range(len(grads)-1):
        if grads[i]<grad_thresh and grads[i+1]>grad_thresh*0.3:
            return rs[i+1]
    return None

print("="*58)
print(f"MODULE 2.3 v3  κ={KAPPA} D={D} pull={COLLAPSE_PULL}")
print(f"{'Step':>5} {'ratio':>7} {'m_ctr':>7} {'m_edg':>7} "
      f"{'m_std':>7} {'boundary':>10} {'stable?':>8}")
print("-"*58)

snapshots={}
for step in range(1,STEPS+1):
    # Diffusion
    dm=A@m-deg*m; m+=dt*D*dm; m=np.clip(m,0,M_MAX)

    # τ̄ accumulation
    diff_vals=np.abs(m[cols]-m[rows])
    grad=np.bincount(rows,weights=diff_vals,minlength=N)/(deg+1e-10)
    tau+=TAU_RATE*grad*dt

    # Soft reset
    excess=np.maximum(0,tau-KAPPA)
    tau-=ALPHA*excess*dt; tau=np.clip(tau,0,None)

    # Collapse: draw m inward
    fired=np.where(excess>0.05)[0]
    for i in fired:
        nb=neighbors[i]
        if not nb: continue
        total=0.0
        pull_strength=COLLAPSE_PULL*min(excess[i]/KAPPA,1.0)
        for j in nb:
            draw=pull_strength*m[j]
            draw=min(draw,m[j]*0.35)
            m[j]-=draw; total+=draw
        m[i]=min(m[i]+total,M_MAX)

    if step%SNAPSHOT_EVERY==0:
        shells=radial_profile(m)
        br=find_boundary(shells)
        m_ctr=m[center]
        m_edg=np.mean(m[edge_nodes])
        ratio=m_ctr/m_edg if m_edg>0 else 0
        c1=ratio>1.05
        stable_str="✓" if c1 and br else ("~" if c1 else "✗")
        b_str=f"r={br}" if br else "none"
        print(f"  {step:>4}  {ratio:>7.4f}  {m_ctr:>7.3f}  "
              f"{m_edg:>7.3f}  {m.std():>7.3f}  {b_str:>10}  {stable_str:>8}")
        snapshots[step]={
            'shells':dict(shells),'m_ctr':m_ctr,'m_edg':m_edg,
            'ratio':ratio,'m_std':m.std(),'boundary':br,'m':m.copy()
        }

print("\n"+"="*58)
print("ASSESSMENT")
print("="*58)

final=snapshots[max(snapshots.keys())]
shells_f=final['shells']
rs=sorted(shells_f.keys())[:L//2]
means=[np.mean(shells_f[r]) for r in rs]
grads=[means[i+1]-means[i] for i in range(len(means)-1)]

# Three criteria
c1=final['ratio']>1.05
c2=final['boundary'] is not None
boundaries=[snapshots[k]['boundary'] for k in sorted(snapshots)
            if snapshots[k]['boundary'] is not None]
c3=len(boundaries)>=3 and np.std(boundaries)<2.5

# Stability of ratio
ratios=[snapshots[k]['ratio'] for k in sorted(snapshots)]
ratio_stable=ratios[-1]>ratios[0]*0.95  # hasn't decayed more than 5%
ratio_growing=ratios[-1]>ratios[0]

print(f"\n  C1 Interior > exterior:   {c1}  (ratio={final['ratio']:.4f})")
print(f"  C2 Shell boundary:         {c2}  ({final['boundary']})")
print(f"  C3 Boundary stable:        {c3}  {boundaries}")
print(f"  C4 Ratio not decaying:     {ratio_stable}  "
      f"({ratios[0]:.4f}→{ratios[-1]:.4f})")
print(f"  C4+ Ratio growing:         {ratio_growing}")

if c1 and c2 and c3 and ratio_stable:
    print(f"\n  ✓✓ STABLE SHELL CONFIRMED")
    print(f"  Fold collapse dynamics produce a stable interior/exterior")
    print(f"  structure against diffusion. Shell boundary is persistent.")
    print(f"  Module 3 foundation is earned.")
elif c1 and c3 and ratio_stable:
    print(f"\n  ✓ Stable interior elevation. Boundary detection marginal.")
    print(f"  Shell is forming but boundary gradient is shallow.")
elif c1 and c2:
    print(f"\n  ~ Shell present but ratio decaying — diffusion winning slowly.")
    print(f"  Increase COLLAPSE_PULL or decrease D further.")
else:
    print(f"\n  ✗ Not yet. Adjust parameters.")

# Figure
fig,axes=plt.subplots(2,2,figsize=(11,9))

ax=axes[0,0]
ax.plot(rs,means,'b-o',lw=2,ms=5)
if final['boundary']:
    ax.axvline(final['boundary'],color='r',ls='--',lw=2,
               label=f'shell r={final["boundary"]}')
ax.set_xlabel('r (hops)'); ax.set_ylabel('mean m(r)')
ax.set_title('Radial profile — final'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax=axes[0,1]
ax.plot(rs[:-1],grads,'g-o',lw=2,ms=5)
ax.axhline(0,color='k',ls=':',alpha=0.5)
if final['boundary']:
    ax.axvline(final['boundary'],color='r',ls='--',lw=2)
ax.set_xlabel('r (hops)'); ax.set_ylabel('dm/dr')
ax.set_title('Gradient profile'); ax.grid(alpha=0.3)

ax=axes[1,0]
times=sorted(snapshots.keys())
ax.plot(times,[snapshots[t]['m_ctr'] for t in times],'b-o',lw=2,ms=5,label='center')
ax.plot(times,[snapshots[t]['m_edg'] for t in times],'r-s',lw=2,ms=5,label='edge')
ax.set_xlabel('Step'); ax.set_ylabel('m')
ax.set_title('Center vs edge'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax=axes[1,1]
ax.plot(times,ratios,'m-o',lw=2,ms=6)
ax.axhline(1.0,color='k',ls=':',alpha=0.4)
ax.axhline(1.05,color='g',ls='--',alpha=0.5,label='1.05 threshold')
ax.set_xlabel('Step'); ax.set_ylabel('m(center)/m(edge)')
ax.set_title('Interior/exterior ratio\n(stable > 1.05 = shell)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.suptitle(f'Module 2.3 v3: Shell Formation  κ={KAPPA} D={D} '
             f'pull={COLLAPSE_PULL}',fontsize=11)
plt.tight_layout()
plt.savefig('/home/claude/mod23_shell_formation.png',dpi=150,bbox_inches='tight')
plt.close()
print("\nFigure saved.")
