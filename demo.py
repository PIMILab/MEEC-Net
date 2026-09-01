"""Minimal demos for MeshlessDEC and MeshlessNeW.

Run directly:
    python demo.py           # Poisson solve
    python demo.py --learn   # Poisson + a short learned-solver training loop
"""

import torch
import torch.nn as nn
import argparse

from meshless_dec import MeshlessDEC, choose_epsilon
from utils import tsolve, tenforce
from model import MeshlessNeW


# notes
# pain points
# MeshlessDEC definition needs to be easier
# need stronger assumptions on how a geometry and input fields are inputted as a data sample
# particularly for boundary data
# model call signature needs to be much clearer for MeshlessNeW
# size determinations should be simpler / automatic
#

# ── helpers ──────────────────────────────────────────────────────────────────

def unit_square_cloud(h):
    """Regular grid point cloud on [0,1]^2 with boundary mask."""
    n = int(round(1.0 / h)) + 1
    x = torch.linspace(0, 1, n, dtype=torch.float64)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    pts = torch.stack([X.ravel(), Y.ravel()], dim=1)
    tol = h / 4
    bd = (pts[:, 0] < tol) | (pts[:, 0] > 1 - tol) | \
         (pts[:, 1] < tol) | (pts[:, 1] > 1 - tol)
    return pts, bd


# ── classical Poisson solve ───────────────────────────────────────────────────

def demo_poisson(h=0.05):
    """Classical meshless Poisson solve:  -∇²p = 1  on [0,1]², p|∂Ω = 0."""
    pts, bd = unit_square_cloud(h)
    eps = choose_epsilon(pts)

    dec = MeshlessDEC(pts, eps, bd)
    dec.assemble(method="dc_pse")

    L   = dec.d0.T @ dec.M1 @ dec.d0
    rhs = dec.M0 @ torch.ones(dec.N, dtype=torch.float64)

    L_bc, rhs_bc = tenforce(L, rhs, D=dec.boundary_idx,
                             x=torch.zeros(len(dec.boundary_idx), dtype=torch.float64))
    p = tsolve(L_bc, rhs_bc)

    print(f"[Poisson]  N={dec.N}  p_max={p.max():.5f}  (exact ≈ 0.0737)")
    return dec, p


# ── learned solver demo ───────────────────────────────────────────────────────

def demo_learn(h=0.05, n_steps=200):
    """Fit MeshlessNeW on a single Poisson instance (illustrative only).

    In practice you would loop over a dataset of different geometries / RHS,
    and keep the DEC assembly outside the training loop for speed.
    """
    dec, p_gt = demo_poisson(h)

    # boundary conditions: p = 0 on all boundary nodes (field 0)
    bcs = [(0, dec.boundary_idx, torch.zeros(len(dec.boundary_idx), dtype=torch.float64))]

    rhs = dec.M0 @ torch.ones(dec.N, dtype=torch.float64)
    z   = dec.pts                   # spatial coords as auxiliary input

    model = MeshlessNeW(z_dim=2, n_fields=1, hidden=32, depth=3)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(n_steps):
        opt.zero_grad()
        p_pred = model(dec, rhs.unsqueeze(1), z, bcs).squeeze(1)
        err    = p_pred - p_gt
        loss   = (err**2 * dec.M0_diag).sum() / (p_gt**2 * dec.M0_diag).sum()
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"  step {step+1:4d}  loss={loss.item():.4e}")

    print(f"[Learn]  final loss={loss.item():.4e}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learn", action="store_true",
                        help="also run the learned solver demo")
    args = parser.parse_args()

    demo_poisson()
    if args.learn:
        demo_learn()
