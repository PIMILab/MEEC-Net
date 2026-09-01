"""Sparse Newton solver with implicit function theorem (IFT) backward.

Exports:
    enforce_bcs, NewtonSolver
    SparseSolve             — custom autograd: x = LU^{-1} rhs
    flux_jac_entries        — per-edge Jacobian via vmap/jacrev
    assemble_sparse_jacobian
    build_picard_laplacian
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


def enforce_bcs(r, u, bcs):
    r = r.clone()
    for dofs, vals in bcs:
        r[dofs] = u[dofs] - vals
    return r


class SparseSolve(torch.autograd.Function):
    """x = LU^{-1} rhs  with IFT backward: dx/d(rhs) = LU^{-T}."""

    @staticmethod
    def forward(ctx, rhs, lu):
        ctx.lu   = lu
        ctx.meta = (rhs.device, rhs.dtype)
        x_np     = lu.solve(rhs.detach().cpu().double().numpy())
        out      = torch.from_numpy(x_np)
        if rhs.device.type == 'cuda':
            out = out.to(dtype=rhs.dtype, device=rhs.device, non_blocking=True)
        return out

    @staticmethod
    def backward(ctx, grad):
        dev, dtype = ctx.meta
        out_np = ctx.lu.solve(grad.detach().cpu().double().numpy(), trans='T')
        out    = torch.from_numpy(out_np)
        if dev.type == 'cuda':
            out = out.to(dtype=dtype, device=dev, non_blocking=True)
        return out, None


class NewtonSolver:
    """Sparse Newton with backtracking line-search and IFT backward pass."""

    def __init__(self, tol=1e-10, maxiter=200, verbose=False, linesearch=True):
        self.tol        = tol
        self.maxiter    = maxiter
        self.verbose    = verbose
        self.linesearch = linesearch
        self.converged  = True
        self.last_rn    = float('inf')
        self.last_iters = 0

    def solve(self, F_raw, bcs, u0, assemble_J):
        u   = u0.clone()
        dev = u.device
        self.converged = True

        def F(u):
            return enforce_bcs(F_raw(u), u, bcs)

        lu         = None
        rn         = float('inf')
        prev_rn    = float('inf')
        stagnation = 0

        for k in range(self.maxiter):
            with torch.no_grad():
                r  = F(u)
                rn = r.norm().item()
            if self.verbose:
                print(f"  Newton {k}: |F|={rn:.4e}")
            if rn < self.tol:
                break
            if k > 10 and rn > 1e6:
                self.converged = False
                break
            if k > 0:
                stagnation = stagnation + 1 if rn >= prev_rn else 0
                if stagnation >= 5:
                    self.converged = False
                    break
            prev_rn = rn

            J    = assemble_J(u)
            lu   = spla.splu(J.tocsc())
            du   = torch.from_numpy(lu.solve((-r).detach().cpu().double().numpy()))
            if dev.type == 'cuda':
                du = du.to(device=dev, dtype=u.dtype, non_blocking=True)

            alpha = 1.0
            if self.linesearch:
                for _ in range(5):
                    with torch.no_grad():
                        if F(u.detach() + alpha * du).norm().item() < rn:
                            break
                    alpha *= 0.5
            u = u.detach() + alpha * du
        else:
            self.converged = False

        self.last_rn    = rn
        self.last_iters = k + 1 if rn >= self.tol else k

        # IFT correction so forward and backward see the same fixed point
        u_star = u.detach()
        if lu is None:
            lu = spla.splu(assemble_J(u_star).tocsc())
        r_diff = F(u_star)
        corr   = torch.from_numpy(lu.solve(r_diff.detach().cpu().double().numpy()))
        if dev.type == 'cuda':
            corr = corr.to(device=dev, dtype=torch.float64)
        u_corrected = u_star - corr
        if not torch.is_grad_enabled():
            return u_corrected
        return u_corrected - SparseSolve.apply(r_diff, lu) + corr


# ── Jacobian assembly ─────────────────────────────────────────────────────

def flux_jac_entries(u_flat, z, N, n_fields, geom, flux_map, edge_z, e_hat=None):
    """Per-edge Jacobians (dfi, dfj) of shape (E, F, F) via vmap/jacrev."""
    from torch.func import jacrev, vmap
    from model import build_features
    F        = n_fields
    U        = u_flat.detach().reshape(F, N).T
    f_in     = torch.cat([U, z.detach()], dim=1)
    need_ehat = flux_map.edge_dirs or bool(flux_map.vec_fields)
    features = build_features(
        f_in, F, flux_map.vec_fields, flux_map.space_dim,
        geom.edges, geom.r,
        edge_z.detach() if edge_z is not None else None,
        e_hat.detach()  if (e_hat is not None and need_ehat) else None,
        edge_dirs=flux_map.edge_dirs,
    ).detach()

    def kernel_row(feat):
        return flux_map.kernel(feat.unsqueeze(0)).squeeze(0)

    dk_all  = vmap(jacrev(kernel_row))(features)   # (E, F, n_feat)
    k_off   = F + z.shape[1]
    dk_mid  = dk_all[:, :, :F]
    hm      = 0.5 * geom.r[:, None, None] * dk_mid
    dk_grad = dk_all[:, :, k_off:k_off + F]
    return hm - dk_grad, hm + dk_grad              # dfi, dfj


def assemble_sparse_jacobian(N, n_fields, geom, eps_vec, gamma,
                              u_flat, z, flux_map, edge_z, flat_bcs, e_hat=None):
    """Sparse (N*F, N*F) Jacobian with BC rows replaced by identity."""
    F  = n_fields
    NF = N * F

    dfi, dfj = flux_jac_entries(u_flat, z, N, F, geom, flux_map, edge_z, e_hat)

    ei    = geom.edges[:, 0].cpu().numpy()
    ej    = geom.edges[:, 1].cpu().numpy()
    m1    = geom.M1_diag.detach().cpu().double().numpy()
    eps_d = np.diag(eps_vec.detach().cpu().double().numpy())
    dfi_t = dfi.detach().cpu().double().numpy().transpose(1, 2, 0)  # (F, F, E)
    dfj_t = dfj.detach().cpu().double().numpy().transpose(1, 2, 0)

    ed = eps_d[:, :, None]
    A  = m1[None, None, :] * (-ed + gamma * dfi_t)
    B  = m1[None, None, :] * ( ed + gamma * dfj_t)

    off   = np.arange(F, dtype=np.int64) * N
    E     = len(ei)
    off_a = off[:, None, None]
    off_b = off[None, :, None]
    ei3   = ei[None, None, :]
    ej3   = ej[None, None, :]
    row_i = np.broadcast_to(off_a + ei3, (F, F, E)).copy().ravel()
    row_j = np.broadcast_to(off_a + ej3, (F, F, E)).copy().ravel()
    col_i = np.broadcast_to(off_b + ei3, (F, F, E)).copy().ravel()
    col_j = np.broadcast_to(off_b + ej3, (F, F, E)).copy().ravel()

    all_rows = np.concatenate([row_i, row_i, row_j, row_j])
    all_cols = np.concatenate([col_i, col_j, col_i, col_j])
    all_vals = np.concatenate([(-A).ravel(), (-B).ravel(), A.ravel(), B.ravel()])

    bc_dofs = (np.concatenate([d.cpu().numpy() for d, _ in flat_bcs])
               if flat_bcs else np.empty(0, dtype=np.int64))
    if len(bc_dofs):
        mask     = np.zeros(NF, dtype=bool)
        mask[bc_dofs] = True
        keep     = ~mask[all_rows]
        all_rows = np.concatenate([all_rows[keep], bc_dofs])
        all_cols = np.concatenate([all_cols[keep], bc_dofs])
        all_vals = np.concatenate([all_vals[keep], np.ones(len(bc_dofs))])

    return sp.coo_matrix((all_vals, (all_rows, all_cols)), shape=(NF, NF)).tocsc()


def build_picard_laplacian(geom, eps_vec, flat_bcs):
    """Block-diagonal eps * L Laplacian for Picard iteration, pre-factorized."""
    N  = geom.N
    F  = len(eps_vec)
    NF = N * F

    d0c   = geom.d0.coalesce()
    ii, jj = d0c.indices()
    d0_sp  = sp.coo_matrix(
        (d0c.values().cpu().double().numpy(),
         (ii.cpu().numpy(), jj.cpu().numpy())),
        shape=(geom.d0.shape[0], N)).tocsr()
    M1_sp = sp.diags(geom.M1_diag.cpu().double().numpy())
    base  = (d0_sp.T @ M1_sp @ d0_sp).tocsr()

    eps_np = eps_vec.cpu().double().numpy()
    L_full = sp.block_diag([float(eps_np[a]) * base for a in range(F)], format='lil')

    bc_dofs = (np.concatenate([d.cpu().numpy() for d, _ in flat_bcs])
               if flat_bcs else np.empty(0, dtype=np.int64))
    for d in bc_dofs:
        L_full[d, :] = 0.0
        L_full[d, d] = 1.0

    return spla.splu(L_full.tocsc())
