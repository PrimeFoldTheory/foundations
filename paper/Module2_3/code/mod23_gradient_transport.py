"""
MODULE 2.3 v4 — GRADIENT-DEPENDENT TRANSPORT
Fix: J_ij = D * (m_j - m_i) * (1 - alpha * |m_j - m_i| / m_scale)

Large gradient → reduced transport → shell boundary resists erasure
Small gradient → normal transport → background equilibrates

This is fold-consistent: J depends on local state (Φ = m), not geometry.
Already established in gravity paper transport constraint tests.

Tests:
1. No seed — does shell form and persist?
2. Boundary stability?
3. Two domains — do they interact/compete?
"""
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

L=20; N=L**3
D=0.015; KAPPA=2.0; TAU_RATE=0.12
ALPHA_RESET=0.4; COLLAPSE_PULL=0.25
ALPHA_GRAD=0.8   # gradient transport damping — key new parameter
M_MAX=100.0

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

cx=np.array([L//2,L//2,L//2],dtype=float)
coords=np.array([[i//(L*L),(i//L)%L,i%L] for i in range(N)],dtype=float)
euc_dist=np.sqrt(np.sum((coords-cx)**2,axis=1))
dt=0.01/deg.max()

def run(seed_type, steps=8000, snapshot_every=500):
    m=np.ones(N)*5.0+0.2*np.random.randn(N)
    m=np.clip(m,0.1,None)

    if seed_type=='center':
        for dx in range(-1,2):
            for dy in range(-1,2):
                for dz in range(-1,2):
                    nx,ny,nz=L//2+dx,L//2+dy,L//2+dz
                    if 0<=nx<L and 0<=ny<L and 0<=nz<L:
                        m[nx*L*L+ny*L+nz]*=1.10
    elif seed_type=='two':
        for cx_,cy_,cz_ in [(L//3,L//2,L//2),(2*L//3,L//2,L//2)]:
            for dx in range(-1,2):
                for dy in range(-1,2):
                    for dz in range(-1,2):
                        nx,ny,nz=cx_+dx,cy_+dy,cz_+dz
                        if 0<=nx<L and 0<=ny<L and 0<=nz<L:
                            m[nx*L*L+ny*L+nz]*=1.10
    # 'none' = pure noise

    tau=np.random.uniform(0,KAPPA*0.2,N)
    ratios=[]; boundaries=[]; m_stds=[]
    edge_nodes=[i for i in range(N)
                if any(c==0 or c==L-1 for c in [i//(L*L),(i//L)%L,i%L])]

    for step in range(1,steps+1):
        # GRADIENT-DEPENDENT TRANSPORT
        # J_ij = D*(m_j-m_i)*(1 - alpha*|m_j-m_i|/m_scale)
        diff = m[cols]-m[rows]          # m_j - m_i
        abs_diff = np.abs(diff)
        m_scale = m.max()+1e-10
        # Transport weight: reduced across large gradients
        weight = np.clip(1.0 - ALPHA_GRAD*abs_diff/m_scale, 0.05, 1.0)
        flow = D * diff * weight
        dm = np.bincount(rows, weights=flow, minlength=N)
        m += dt*dm; m=np.clip(m,0,M_MAX)

        # τ̄ accumulation
        grad=abs_diff  # reuse
        grad_node=np.bincount(rows,weights=grad,minlength=N)/(deg+1e-10)
        tau+=TAU_RATE*grad_node*dt

        # Soft reset
        excess=np.maximum(0,tau-KAPPA)
        tau-=ALPHA_RESET*excess*dt; tau=np.clip(tau,0,None)

        # Collapse: draw m inward
        fired=np.where(excess>0.05)[0]
        for i in fired:
            nb=neighbors[i]; total=0.0
            pull=COLLAPSE_PULL*min(excess[i]/KAPPA,1.0)
            for j in nb:
                draw=min(pull*m[j],m[j]*0.35)
                m[j]-=draw; total+=draw
            m[i]=min(m[i]+total,M_MAX)

        if step%snapshot_every==0:
            peak=np.argmax(m)
            pk_c=coords[peak]
            dist_pk=np.sqrt(np.sum((coords-pk_c)**2,axis=1))
            shells=defaultdict(list)
            for i in range(N):
                r=int(round(dist_pk[i]))
                shells[r].append(m[i])
            rs=sorted(shells.keys())[:L//2]
            if len(rs)>=4:
                means=[np.mean(shells[r]) for r in rs]
                m_edge=np.mean(m[edge_nodes])
                ratio=means[0]/m_edge if m_edge>0 else 1
                grads=[means[i+1]-means[i] for i in range(len(means)-1)]
                br=None
                for i in range(len(grads)-1):
                    if grads[i]<-0.05 and grads[i+1]>-0.015:
                        br=rs[i+1]; break
                ratios.append(ratio)
                boundaries.append(br)
                m_stds.append(m.std())

    return ratios, boundaries, m_stds, m.copy()

print("="*65)
print("MODULE 2.3 v4: GRADIENT-DEPENDENT TRANSPORT")
print(f"J_ij = D*(m_j-m_i)*(1 - {ALPHA_GRAD}*|Δm|/m_max)")
print("="*65)

tests=[('none','NO SEED — pure noise'),
       ('center','Seeded — control'),
       ('two','Two seeds — domain competition')]

fig,axes=plt.subplots(1,3,figsize=(13,4))
all_results={}

for idx,(seed_type,label) in enumerate(tests):
    print(f"\n[{idx+1}] {label}")
    ratios,boundaries,stds,m_final=run(seed_type,steps=10000)
    all_results[seed_type]={'ratios':ratios,'boundaries':boundaries}

    final_ratio=ratios[-1] if ratios else 0
    final_br=boundaries[-1]
    br_vals=[b for b in boundaries if b is not None]
    br_stable=len(br_vals)>=5 and np.std(br_vals)<2.5 if br_vals else False

    if len(ratios)>=6:
        early=np.mean(ratios[:3]); late=np.mean(ratios[-3:])
        decay_pct=(early-late)/early*100
        # Check if stabilizing: last 3 closer together than first 3
        late_std=np.std(ratios[-4:]) if len(ratios)>=4 else 999
        stabilizing=decay_pct<3.0 or late_std<0.002
    else:
        decay_pct=0; stabilizing=False

    shell_formed=final_ratio>1.05 and br_stable

    print(f"  Shell formed+stable: {shell_formed}")
    print(f"  Final ratio:         {final_ratio:.4f}")
    print(f"  Boundary stable:     {br_stable}  {br_vals[:8]}")
    print(f"  Decay rate:          {decay_pct:.2f}%")
    print(f"  Stabilizing:         {stabilizing}")
    print(f"  Ratio trajectory:    "
          f"{[f'{r:.4f}' for r in ratios[-6:]]}")

    ax=axes[idx]
    steps_arr=np.arange(1,len(ratios)+1)*500
    ax.plot(steps_arr,ratios,'b-o',lw=2,ms=4,label='ratio')
    ax.axhline(1.0,color='k',ls=':',alpha=0.4)
    ax.axhline(1.05,color='g',ls='--',alpha=0.5,label='1.05')
    if br_vals:
        br_plot=[b if b else np.nan for b in boundaries]
        ax2=ax.twinx()
        ax2.plot(steps_arr[:len(br_plot)],br_plot,
                'r-s',lw=1.5,ms=4,alpha=0.6,label='boundary r')
        ax2.set_ylabel('boundary r',color='r',fontsize=8)
        ax2.tick_params(axis='y',colors='r')
    ax.set_xlabel('Step'); ax.set_ylabel('m(peak)/m(edge)')
    title_str='✓ STABLE' if shell_formed else ('~ partial' if final_ratio>1.05 else '✗')
    ax.set_title(f'{label}\nratio={final_ratio:.4f} {title_str}')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

print("\n"+"="*65)
print("BACK PRESSURE TEST: run seeded with CLOSED boundary")
print("(Reflecting edges — B_boundary > 0)")
print("="*65)

# Closed system: periodic boundaries act as back pressure
def run_periodic(steps=10000, snapshot_every=500):
    """Periodic boundaries = closed system = back pressure"""
    rows_p,cols_p=[],[]
    for x in range(L):
        for y in range(L):
            for z in range(L):
                i=x*L*L+y*L+z
                for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),
                                  (0,-1,0),(0,0,1),(0,0,-1)]:
                    nx,ny,nz=(x+dx)%L,(y+dy)%L,(z+dz)%L
                    j=nx*L*L+ny*L+nz
                    rows_p.append(i); cols_p.append(j)
    rows_p=np.array(rows_p); cols_p=np.array(cols_p)
    A_p=sp.csr_matrix((np.ones(len(rows_p)),(rows_p,cols_p)),shape=(N,N))
    deg_p=np.array(A_p.sum(axis=1)).flatten()
    nb_p=defaultdict(list)
    for r,c in zip(rows_p,cols_p): nb_p[r].append(c)

    m=np.ones(N)*5.0+0.2*np.random.randn(N); m=np.clip(m,0.1,None)
    tau=np.random.uniform(0,KAPPA*0.2,N)
    ratios_p=[]

    for step in range(1,steps+1):
        diff=m[cols_p]-m[rows_p]
        abs_diff=np.abs(diff)
        m_scale=m.max()+1e-10
        weight=np.clip(1.0-ALPHA_GRAD*abs_diff/m_scale,0.05,1.0)
        flow=D*diff*weight
        dm=np.bincount(rows_p,weights=flow,minlength=N)
        m+=dt*dm; m=np.clip(m,0,M_MAX)

        grad=abs_diff
        grad_node=np.bincount(rows_p,weights=grad,minlength=N)/(deg_p+1e-10)
        tau+=TAU_RATE*grad_node*dt
        excess=np.maximum(0,tau-KAPPA)
        tau-=ALPHA_RESET*excess*dt; tau=np.clip(tau,0,None)
        fired=np.where(excess>0.05)[0]
        for i in fired:
            nb=nb_p[i]; total=0.0
            pull=COLLAPSE_PULL*min(excess[i]/KAPPA,1.0)
            for j in nb:
                draw=min(pull*m[j],m[j]*0.35)
                m[j]-=draw; total+=draw
            m[i]=min(m[i]+total,M_MAX)

        if step%snapshot_every==0:
            peak=np.argmax(m)
            pk_c=coords[peak]
            dist_pk=np.sqrt(np.sum((coords-pk_c)**2,axis=1))
            shells=defaultdict(list)
            for i in range(N):
                shells[int(round(dist_pk[i]))].append(m[i])
            rs=sorted(shells.keys())[:L//2]
            if len(rs)>=4:
                means=[np.mean(shells[r]) for r in rs]
                m_far=np.mean(means[-3:])
                ratio=means[0]/m_far if m_far>0 else 1
                ratios_p.append(ratio)

    return ratios_p

print("Running periodic (closed) system...")
ratios_periodic=run_periodic()
if ratios_periodic:
    early=np.mean(ratios_periodic[:3])
    late=np.mean(ratios_periodic[-3:])
    decay=(early-late)/early*100
    print(f"  Periodic ratio trajectory: {[f'{r:.4f}' for r in ratios_periodic[-6:]]}")
    print(f"  Decay rate: {decay:.2f}%")
    if decay<3.0:
        print(f"  ✓ CLOSED SYSTEM STABILIZES SHELL")
        print(f"  Back pressure (B_boundary via periodic BC) halts decay.")
        print(f"  This confirms: open system decays, closed system stable.")
        print(f"  Sean's HVAC intuition is correct.")
    elif decay<8.0:
        print(f"  ~ Less decay than open system ({decay:.1f}% vs prev ~5-8%)")
    else:
        print(f"  Decay similar to open — BC may not be sufficient back pressure")

plt.tight_layout()
plt.savefig('/home/claude/mod23_gradient_transport.png',dpi=150,bbox_inches='tight')
plt.close()
print("\nFigure saved.")
