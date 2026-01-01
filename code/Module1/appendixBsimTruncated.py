#!/usr/bin/env python3
"""
fold_soc_1d.py

Minimal 1D Fold-inspired SOC toy.
State per site i:
  - phi[i]  : Phi (Fold-Density / load)   [kept constant in the minimal test]
  - tau[i]  : tau_bar (Fold-Time / tension accumulator)
  - kappa_i : kappa (Threshold / collapse bound), with small disorder to break sync

Dynamics:
  Slow drive: pick a random site i, add drive_amount/phi[i] to tau[i]
  Fast relaxation (avalanche): if phi[i]*tau[i] >= kappa_i[i], topple:
     drop = kappa_i[i]/phi[i]
     tau[i] -= drop   (NOT full reset; this is sandpile-like "exact threshold drop")
     distribute drop/2 to neighbors, boundaries dissipate into sink ledger S

Notes:
- This is "illustrative / non-normative": it demonstrates operational semantics and cascades.
- If you want strict "reset-to-zero" semantics, see RESET_MODE below.
"""

import csv
import os
import time
import random
from dataclasses import dataclass, asdict
from collections import deque
from typing import List, Tuple, Dict, Optional
import subprocess #----------------------
import socket     #####for provenance key
import uuid       #----------------------

import numpy as np

#---------------------------
# provenance key
#---------------------------
def get_git_info():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL
        ) != 0
        return commit, dirty
    except Exception:
        return "nogit", False

# -----------------------------
# Configuration
# -----------------------------

@dataclass
class FoldSOCConfig:
    N: int = 128
    steps: int = 200_000
    warmup: int = 10_000
    drive_amount: float = 1.0
    kappa: float = 10.0
    kappa_jitter: float = 0.05     # ±5% disorder (breaks synchronization)
    seed: int = 1

    # Toppling mode:
    # "drop"  = subtract exactly kappa_i/phi (sandpile-like)
    # "reset" = reset tau[i] to 0.0 at collapse (Fold-Time truncation style)
    topple_mode: str = "drop"

    # Boundary handling:
    # "sink"  = dissipate off-lattice flow into sink ledger S
    boundary_mode: str = "sink"

    # Output
    out_dir: str = "truncated_fold_runs"
    tag: str = "toy"


# -----------------------------
# Core simulation
# -----------------------------

def run_fold_soc_1d(cfg: FoldSOCConfig) -> Tuple[np.ndarray, float, Dict[str, float]]:
    # Repro: seed both RNGs
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    N = cfg.N

    # Phi (Fold-Density / load): constant for minimal test
    phi = np.ones(N, dtype=float)

    # tau_bar (Fold-Time / tension accumulator)
    tau = np.zeros(N, dtype=float)

    # Sink ledger S
    sink = 0.0

    # Per-site thresholds kappa_i with disorder
    # kappa_i = kappa * (1 ± jitter)
    kappa_i = np.array([
        cfg.kappa * (1.0 + cfg.kappa_jitter * (2.0 * random.random() - 1.0))
        for _ in range(N)
    ], dtype=float)

    def unstable(i: int) -> bool:
        # Site i fires when Phi[i] * tau[i] crosses kappa_i
        return (phi[i] * tau[i]) >= kappa_i[i]

    avalanche_sizes: List[int] = []

    # sanity counters
    events = 0
    total_topples = 0
    max_queue = 0

    t0 = time.time()

    for t in range(cfg.steps + cfg.warmup):

        # ---- Slow drive (Anti-Zeno style input) ----
        i = random.randrange(N)
        tau[i] += cfg.drive_amount / phi[i]

        # ---- Fast relaxation (avalanche) ----
        size = 0
        q = deque([i])
        inq = {i}

        while q:
            j = q.pop()
            inq.discard(j)
            if not unstable(j):
                continue

            size += 1
            total_topples += 1

            if cfg.topple_mode == "drop":
                # subtract exactly threshold-worth of tau (in tau-units)
                drop = kappa_i[j] / phi[j]
                tau[j] -= drop
                # numerical floor (avoid tiny negatives)
                if tau[j] < 0.0:
                    tau[j] = 0.0

            elif cfg.topple_mode == "reset":
                # Fold-Time truncation semantics: tau -> 0 on collapse
                drop = tau[j]
                tau[j] = 0.0

            else:
                raise ValueError(f"Unknown topple_mode: {cfg.topple_mode}")

            # redistribute "drop" equally to neighbors
            left_share = 0.5 * drop
            right_share = 0.5 * drop

            # left neighbor or boundary sink
            if j - 1 >= 0:
                tau[j - 1] += left_share
                if unstable(j - 1) and (j - 1) not in inq:
                    q.append(j - 1); inq.add(j - 1)
            else:
                if cfg.boundary_mode == "sink":
                    sink += left_share
                else:
                    # reflect (optional alt): put it back on site j
                    tau[j] += left_share

            # right neighbor or boundary sink
            if j + 1 < N:
                tau[j + 1] += right_share
                if unstable(j + 1) and (j + 1) not in inq:
                    q.append(j + 1); inq.add(j + 1)
            else:
                if cfg.boundary_mode == "sink":
                    sink += right_share
                else:
                    tau[j] += right_share

            if len(q) > max_queue:
                max_queue = len(q)

            # ---- Sanity checks (Pass #4) ----
            # Phi constant nonnegative
            assert np.all(phi >= 0.0), "Phi went negative (should not happen)."
            # Tau must stay nonnegative
            assert np.all(tau >= 0.0), "Tau went negative (check redistribution/topple)."

        if t >= cfg.warmup:
            avalanche_sizes.append(size)
            if size > 0:
                events += 1

        # ---- progress heartbeat ----
        if (t % 5000) == 0 and t > 0:
            done = 100.0 * t / (cfg.steps + cfg.warmup)
            rate = t / (time.time() - t0)
            print(
                f"[N={cfg.N}] {done:5.1f}%  t={t:,}/{cfg.steps+cfg.warmup:,}  "
                f"events={events:,}  topples={total_topples:,}  rate={rate:,.0f} steps/s",
                flush=True
            )
        

    dt = time.time() - t0
    sizes = np.asarray(avalanche_sizes, dtype=int)

    summary = {
        "events": float(events),
        "event_fraction": float((sizes > 0).mean()),
        "max_avalanche": float(sizes.max()) if sizes.size else 0.0,
        "mean_avalanche_nonzero": float(sizes[sizes > 0].mean()) if np.any(sizes > 0) else 0.0,
        "total_topples": float(total_topples),
        "max_queue": float(max_queue),
        "runtime_sec": float(dt),
    }
    return sizes, float(sink), summary


# -----------------------------
# Stats helpers
# -----------------------------

def print_stats(sizes: np.ndarray, sink: float, summary: Dict[str, float]) -> None:
    s = sizes
    nz = s[s > 0]

    print("total recorded steps:", len(s))
    print("nonzero avalanches:", int((s > 0).sum()))
    print("fraction nonzero:", float((s > 0).mean()))
    print("max avalanche:", int(nz.max()) if nz.size else 0)
    print("sink ledger S:", sink)

    if nz.size:
        print("p50:", int(np.percentile(nz, 50)))
        print("p90:", int(np.percentile(nz, 90)))
        print("p99:", int(np.percentile(nz, 99)))

    print("total topples:", int(summary.get("total_topples", 0)))
    print("runtime_sec:", summary.get("runtime_sec", 0.0))


def ccdf_snapshot(sizes: np.ndarray, bins: int = 12) -> List[Tuple[int, float]]:
    """Return a small CCDF snapshot: (size, P(S>=size)) points."""
    nz = sizes[sizes > 0]
    if nz.size == 0:
        return []
    vals, counts = np.unique(nz, return_counts=True)
    ccdf = 1.0 - (np.cumsum(counts) / np.sum(counts))
    step = max(1, len(vals) // bins)
    out = []
    for v, c in zip(vals[::step], ccdf[::step]):
        out.append((int(v), float(c)))
    return out


def mle_alpha(data: np.ndarray, xmin: float) -> Tuple[Optional[float], int]:
    """Rough MLE exponent for a power-law tail. For sanity only."""
    tail = data[data >= xmin]
    if tail.size < 200:
        return None, int(tail.size)
    alpha = 1.0 + (tail.size / np.sum(np.log(tail / xmin)))
    return float(alpha), int(tail.size)


# -----------------------------
# Output (CSV)
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_run_csv(cfg: FoldSOCConfig, sizes: np.ndarray, sink: float, summary: Dict[str, float]) -> str:
    ensure_dir(cfg.out_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{cfg.tag}_N{cfg.N}_a{cfg.drive_amount}_k{cfg.kappa}_j{cfg.kappa_jitter}_seed{cfg.seed}_{stamp}.csv"
    fpath = os.path.join(cfg.out_dir, fname)

   # -------- provenance key (repo fingerprint) --------
    run_id = stamp + "_" + uuid.uuid4().hex[:8]
    git_commit, git_dirty = get_git_info()
    host = socket.gethostname()
    # ---------------------------------------------------

    with open(fpath, "w", newline="") as f:
        w = csv.writer(f)
        # header metadata
        w.writerow(["# config"])
        
        # provenance key lines (go right under "# config")
        w.writerow(["# run_id", run_id])
        w.writerow(["# git_commit", git_commit])
        w.writerow(["# git_dirty", git_dirty])
        w.writerow(["# host", host])
        
        for k, v in asdict(cfg).items():
            w.writerow([f"# {k}", v])
        w.writerow(["# summary"])
        for k, v in summary.items():
            w.writerow([f"# {k}", v])
        w.writerow(["# data"])
        w.writerow(["t", "avalanche_size"])
        for t, sz in enumerate(sizes):
            w.writerow([t, int(sz)])

    return fpath


# -----------------------------
# Main: multi-N sweep (small)
# -----------------------------

if __name__ == "__main__":
    # Example small sweep for appendix-table rows
    # Keep it modest: a few runs, a few Ns.
    for N in [64, 128, 256]:
        cfg = FoldSOCConfig(
            N=N,
            steps=200_000,
            warmup=10_000,
            drive_amount=1.0,
            kappa=10.0,
            kappa_jitter=0.05,
            seed=1,
            topple_mode="reset",        # try "reset" if you want strict truncation semantics
            boundary_mode="sink",
            out_dir="truncated_fold_runs",
            tag="foldtoy"
        )

        sizes, sink, summary = run_fold_soc_1d(cfg)

        print("\n==============================")
        print(f"=== N = {N} ===")
        print("mode:", cfg.topple_mode, "| boundary:", cfg.boundary_mode)
        print_stats(sizes, sink, summary)

        # CCDF snapshot (small)
        snap = ccdf_snapshot(sizes, bins=10)
        if snap:
            print("\nCCDF snapshot (size, P(S>=size)):")
            for v, p in snap:
                print(v, p)

        # Tail meter
        nz = sizes[sizes > 0].astype(float)
        print("\nTail meter (rough MLE exponent):")
        for xmin in [50, 100, 200, 500, 1000, 2000]:
            a, n = mle_alpha(nz, xmin)
            print(f"xmin={xmin:>4}  alpha={a}  tail_n={n}")

        # Save CSV for reproducibility + appendix sourcing
        out = write_run_csv(cfg, sizes, sink, summary)
        print("\nSaved run CSV:", out)
