import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import scipy.sparse.linalg as spla

from solver import (SparseSolve, NewtonSolver, enforce_bcs,
                    assemble_sparse_jacobian, build_picard_laplacian)

# ── kernel utilities ──────────────────────────────────────────────────────

class LipLinear(nn.Module):
    """Linear layer with per-row L1 Lipschitz constraint."""
    def __init__(self, d_in, d_out, lip=1.0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self.bias   = nn.Parameter(torch.zeros(d_out)) if bias else None
        self.lip    = lip
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x):
        row_norm = self.weight.abs().sum(1, keepdim=True).clamp(min=1e-8)
        scale    = (self.lip / row_norm).clamp(max=1.0)
        return F_torch.linear(x, self.weight * scale, self.bias)


def make_kernel(n_in, n_out=1, hidden=32, n_layers=2, lip=None):
    """MLP  R^{n_in} -> R^{n_out}."""
    def Linear(d_in, d_out, bias=True):
        if lip is not None:
            per = lip ** (1.0 / (n_layers + 1))
            return LipLinear(d_in, d_out, per, bias)
        return nn.Linear(d_in, d_out, bias=bias)

    layers, d = [], n_in
    for _ in range(n_layers):
        layers += [Linear(d, hidden), nn.Tanh()]
        d = hidden
    layers.append(Linear(d, n_out, bias=False))
    return nn.Sequential(*layers)


# ── pullback / contraction ────────────────────────────────────────────────

def pullback(f, edges, h):
    """(N,k) nodal -> (E,k) midpoint and (E,k) directional gradient."""
    fi, fj = f[edges[:, 0]], f[edges[:, 1]]
    return 0.5 * (fi + fj), (fj - fi) / h.unsqueeze(1)


def contract_vector(v, edges, e_hat):
    """(N,d) nodal vector -> (E,1) 1-form via midpoint projection."""
    v_mid = 0.5 * (v[edges[:, 0]] + v[edges[:, 1]])
    return (v_mid * e_hat).sum(-1, keepdim=True)


def contract_tensor(T, edges, e_hat):
    """(N,d,d) symmetric tensor -> (E,1) via midpoint + eᵀTe."""
    T_mid = 0.5 * (T[edges[:, 0]] + T[edges[:, 1]])
    return torch.einsum('ea,eab,eb->e', e_hat, T_mid, e_hat).unsqueeze(1)


# ── feature assembly ──────────────────────────────────────────────────────

# TODO unused shape arguments?
def build_features(f, n_fields, vec_fields, space_dim,
                   edges, h,
                   edge_z=None, e_hat=None, edge_dirs=False):
    """Per-edge kernel features: [mid | grad | transv_v... | edge_z | e_hat].

    mid      = (f_i+f_j)/2                          (k)
    grad     = (f_j-f_i)/h                          (k)
    transv_v = grad[:,indices] - along-edge part     (d_v per vector group)
    edge_z   = contracted auxiliary 1-forms          (p, optional)
    e_hat    = unit edge tangent                     (d, if edge_dirs)
    """
    # fi, fj = f[edges[:, 0]], f[edges[:, 1]]
    # mid  = 0.5 * (fi + fj)
    # grad = (fj - fi) / h.unsqueeze(1)
    parts = pullback(f, edges, h)

    # parts = [mid, grad]

    # Transverse features for declared vector field groups
    if vec_fields and e_hat is not None:
        for indices in vec_fields:
            du       = parts[1][:, indices]                             # (E, d_v)
            du_along = (du * e_hat).sum(-1, keepdim=True) * e_hat  # (E, d_v)
            parts.append(du - du_along)                            # (E, d_v)  perp to e_hat

    if edge_z is not None:
        parts.append(edge_z)
    if edge_dirs and e_hat is not None:
        parts.append(e_hat)
    return torch.cat(parts, dim=1)


# ── learned flux map ──────────────────────────────────────────────────────

class CochainMap(nn.Module):
    """Learned 1-form: features -> kernel -> h * output."""

    def __init__(self, kernel, edge_dirs=False, vec_fields=None, space_dim=2):
        super().__init__()
        self.kernel     = kernel
        self.edge_dirs  = edge_dirs
        self.vec_fields = vec_fields or []
        self.space_dim  = space_dim

    def forward(self, f, n_fields, edges, h, edge_z=None, e_hat=None):
        # feature pullback, kernel application and line integration
        feat = build_features(
            f, n_fields, self.vec_fields, self.space_dim,
            edges, h,
            edge_z, e_hat,
            edge_dirs=self.edge_dirs,
        )
        return h.unsqueeze(1) * self.kernel(feat)


def assemble_sparse_jacobian(N, n_fields, geom, eps_vec, gamma,
                              u_flat, z, flux_map, edge_z, flat_bcs,
                              e_hat=None):
    """Sparse (N*F, N*F) Jacobian with BC rows replaced by identity rows."""
    F  = n_fields
    NF = N * F

    dfi, dfj = flux_jac_entries(u_flat, z, N, F, geom, flux_map, edge_z,
                                e_hat)

    ei     = geom.edges[:, 0].cpu().numpy()
    ej     = geom.edges[:, 1].cpu().numpy()
    m1     = geom.M1_diag.detach().cpu().double().numpy()
    eps_d  = np.diag(eps_vec.detach().cpu().double().numpy())   # (F, F)
    dfi_t  = dfi.detach().cpu().double().numpy().transpose(1, 2, 0)  # (F, F, E)
    dfj_t  = dfj.detach().cpu().double().numpy().transpose(1, 2, 0)

    ed = eps_d[:, :, None]
    A  = m1[None, None, :] * (-ed + gamma * dfi_t)
    B  = m1[None, None, :] * ( ed + gamma * dfj_t)

    off   = np.arange(F, dtype=np.int64) * N
    off_a = off[:, None, None]
    off_b = off[None, :, None]
    E     = len(ei)
    ei3   = ei[None, None, :]
    ej3   = ej[None, None, :]
    row_i = np.broadcast_to(off_a + ei3, (F, F, E)).copy().ravel()
    row_j = np.broadcast_to(off_a + ej3, (F, F, E)).copy().ravel()
    col_i = np.broadcast_to(off_b + ei3, (F, F, E)).copy().ravel()
    col_j = np.broadcast_to(off_b + ej3, (F, F, E)).copy().ravel()

    all_rows = np.concatenate([row_i, row_i, row_j, row_j])
    all_cols = np.concatenate([col_i, col_j, col_i, col_j])
    all_vals = np.concatenate([(-A).ravel(), (-B).ravel(), A.ravel(), B.ravel()])

    # BC rows: filter out existing entries, add identity diagonal
    if flat_bcs:
        bc_dofs = np.concatenate([d.cpu().numpy() for d, _ in flat_bcs])
    else:
        bc_dofs = np.empty(0, dtype=np.int64)
    if len(bc_dofs):
        bc_set   = np.zeros(NF, dtype=bool)
        bc_set[bc_dofs] = True
        keep     = ~bc_set[all_rows]
        all_rows = np.concatenate([all_rows[keep], bc_dofs])
        all_cols = np.concatenate([all_cols[keep], bc_dofs])
        all_vals = np.concatenate([all_vals[keep], np.ones(len(bc_dofs))])

# ── Picard Laplacian builder ─────────────────────────────────────────────

def build_picard_laplacian(geom, eps_vec, flat_bcs):
    """Build and factorize the NF×NF block-diagonal Laplacian for Picard iteration.

    Block a = eps_a * d0^T diag(M1) d0.  BC rows replaced with identity rows.
    Returns scipy SuperLU object ready for .solve().
    """
    N  = geom.N
    F  = len(eps_vec)
    NF = N * F

    # Scipy d0 from torch sparse
    d0c  = geom.d0.coalesce()
    ii, jj = d0c.indices()
    vv   = d0c.values().cpu().double().numpy()
    E    = geom.d0.shape[0]
    d0_sp = sp.coo_matrix((vv, (ii.cpu().numpy(), jj.cpu().numpy())),
                          shape=(E, N)).tocsr()
    M1_sp = sp.diags(geom.M1_diag.cpu().double().numpy())
    base  = (d0_sp.T @ M1_sp @ d0_sp).tocsr()  # (N, N) base Laplacian

    eps_np = eps_vec.cpu().double().numpy()  # (F,)

    # Build list of per-field blocks
    blocks = [float(eps_np[a]) * base for a in range(F)]
    L_full = sp.block_diag(blocks, format='lil')  # (NF, NF)

    # Replace BC rows with identity
    if flat_bcs:
        bc_dofs = np.concatenate([d.cpu().numpy() for d, _ in flat_bcs])
    else:
        bc_dofs = np.empty(0, dtype=np.int64)
    for d in bc_dofs:
        L_full[d, :] = 0.0
        L_full[d, d] = 1.0

    return spla.splu(L_full.tocsc())


# ── main model ─────────────────────────────────────────────────────────────

class MeshlessNeW(nn.Module):
    """Meshless Neural Whitney Forms solver.

    Solves: d0^T M1 (eps_a * d0 u_a + gamma * K_a(u, z)) = rhs_a

    z   : (N, z_dim)  auxiliary nodal inputs (e.g. coordinates, parameters)
    bcs : list of (field_idx, node_indices, values)
    """

    def __init__(self, n_fields=1, z_dim=0, z_v_dim=0, z_t_dim=0, z_edge_dim=0,
                 hidden=128, n_layers=2,
                 gamma=1.0, eps=1.0, lip=None, edge_dirs=False,
                 vec_fields=None, space_dim=2,
                 solver_type='newton', field_scale=None,
                 tol=1e-10, maxiter=200, verbose=False, linesearch=True,
                 source_model=None):
        super().__init__()
        self.n_fields     = n_fields
        self.edge_dirs    = edge_dirs
        self.gamma        = gamma
        self.source_model = source_model
        self.tol          = tol
        self.maxiter      = maxiter

        if vec_fields is None:
            vec_fields = []
        for g in vec_fields:
            assert len(g) == space_dim, \
                f"vec_fields group {g} has {len(g)} components but space_dim={space_dim}"
            assert all(0 <= i < n_fields for i in g), \
                f"vec_fields group {g} has indices outside [0, {n_fields})"
        self.vec_fields = vec_fields
        self.space_dim  = space_dim

        if field_scale is None:
            fs = torch.ones(n_fields, dtype=torch.float64)
        else:
            fs = torch.tensor(field_scale, dtype=torch.float64)
        self.register_buffer('field_scale', fs)

        k     = n_fields + z_dim
        n_in  = 2 * k + int(z_v_dim) + int(z_t_dim) + int(z_edge_dim)
        n_in += sum(len(g) for g in vec_fields)
        if edge_dirs:
            n_in += space_dim

        kernel = make_kernel(n_in=n_in, n_out=n_fields, hidden=hidden,
                             n_layers=n_layers, lip=lip)
        self.flux_map = CochainMap(kernel, edge_dirs=edge_dirs,
                                   vec_fields=vec_fields, space_dim=space_dim)

        if isinstance(eps, (int, float)):
            eps = [eps] * n_fields
        self.register_buffer('eps_vec', torch.tensor(eps, dtype=torch.float64))

        self.solver_type = solver_type
        self.solver = NewtonSolver(tol=tol, maxiter=maxiter, verbose=verbose,
                                   linesearch=linesearch)

    def edge_features(self, geom, z_v=None, z_t=None, z_edge=None):
        """Contract optional vector/tensor z fields to edge 1-forms."""
        parts = []
        e_hat = geom.dx / geom.r.unsqueeze(1)
        if z_v is not None:
            parts.append(contract_vector(z_v, geom.edges, e_hat))
        if z_t is not None:
            parts.append(contract_tensor(z_t, geom.edges, e_hat))
        if z_edge is not None:
            parts.append(z_edge)
        return torch.cat(parts, dim=1) if parts else None

    def get_rhs(self, geom, rhs=None, z=None):
        if rhs is not None:
            return rhs
        if self.source_model is not None:
            return geom.M0_diag.unsqueeze(1) * self.source_model(z, geom)
        return torch.zeros(geom.N, self.n_fields,
                           dtype=geom.pts.dtype, device=geom.pts.device)

    def prepare(self, geom, bcs, rhs=None):
        """Precompute static solve inputs (cacheable across training steps).
        BCs and RHS are stored in normalized (scaled) space.
        """
        N, F     = geom.N, self.n_fields
        fs       = self.field_scale.to(geom.pts.device)   # (F,)
        rhs_raw  = self.get_rhs(geom, rhs)                # (N, F) physical
        rhs_flat = (rhs_raw / fs).T.reshape(-1)           # normalized: rhs / scale
        # initial guess always all interior zeros, and BCs on boundary nodes
        u0       = torch.zeros(N, F, dtype=geom.pts.dtype, device=geom.pts.device)
        for field, nodes, vals in bcs:
            u0[nodes, field] = vals / fs[field]           # normalize
        return dict(u0=u0.T.reshape(-1).detach(), rhs=rhs_flat)

    def residual(self, u_flat, z, geom, rhs_flat, edge_z=None, e_hat=None):
        N, F    = geom.N, self.n_fields
        U       = u_flat.reshape(F, N).T                     # (N, F)
        flux    = self.flux_map(torch.cat([U, z], dim=1),
                                N, geom.edges, geom.r,
                                edge_z=edge_z, e_hat=e_hat)  # (E, F)
        d0U     = geom.d0 @ U                                # (E, F)
        sigma   = self.eps_vec[None, :] * d0U + self.gamma * flux   # (E, F)
        div_sig = geom.d0.T @ (geom.M1_diag[:, None] * sigma)       # (N, F)
        return (div_sig - rhs_flat.reshape(F, N).T).T.reshape(-1)    # (N*F,)

    # TODO find somewhere cleaner to put this, maybe in solver.py
    def _picard(self, z, geom, flat_bcs, u0_flat, rhs_flat, edge_z, e_hat):
        """Picard iteration:  L_eps U^{k+1} = rhs - gamma * d0^T M1 K(U^k).

        L_eps is factorized once; only forward passes of flux_map per iteration.
        IFT backward uses the same SparseSolve as Newton.
        """
        N, F = geom.N, self.n_fields
        dev  = u0_flat.device
        sol  = self.solver  # reuse convergence bookkeeping
        sol.converged, sol.last_rn = True, float('inf')

        lu_L = build_picard_laplacian(geom, self.eps_vec, flat_bcs)

        # Precompute BC value vector for RHS replacement
        bc_rhs = torch.zeros(N * F, dtype=torch.float64, device=dev)
        if flat_bcs:
            for dofs, vals in flat_bcs:
                bc_rhs[dofs] = vals
        bc_mask = torch.zeros(N * F, dtype=torch.bool, device=dev)
        if flat_bcs:
            bc_mask[torch.cat([d for d, _ in flat_bcs])] = True

        u = u0_flat.clone()
        rn = float('inf')
        for k in range(sol.maxiter):
            with torch.no_grad():
                r  = enforce_bcs(
                         self.residual(u, z, geom, rhs_flat, edge_z, e_hat),
                         u, flat_bcs)
                rn = r.norm().item()
            if sol.verbose:
                print(f"  Picard {k}: |F|={rn:.4e}")
            if rn < sol.tol:
                break
            if k > 5 and rn > 1e6:
                sol.converged = False
                break

            # Build RHS: rhs_flat - gamma * d0^T M1 K(u^k),  then pin BC dofs
            with torch.no_grad():
                U    = u.reshape(F, N).T
                flux = self.flux_map(torch.cat([U, z], dim=1),
                                     N, geom.edges, geom.r,
                                     edge_z=edge_z, e_hat=e_hat)  # (E, F)
                div_K = geom.d0.T @ (geom.M1_diag[:, None] * flux)  # (N, F)
                rhs_k = rhs_flat - self.gamma * div_K.T.reshape(-1)  # (NF,)
                rhs_k[bc_mask] = bc_rhs[bc_mask]                     # Dirichlet

            rhs_np = rhs_k.detach().cpu().double().numpy()
            u_np   = lu_L.solve(rhs_np)
            u      = torch.from_numpy(u_np)
            if dev.type == 'cuda':
                u = u.to(device=dev, dtype=torch.float64)
        else:
            sol.converged = False

        sol.last_rn    = rn
        sol.last_iters = k + 1

        # IFT: one Newton correction so train/eval return the same value.
        u_star = u.detach()
        J      = assemble_sparse_jacobian(N, F, geom, self.eps_vec, self.gamma,
                                          u_star, z, self.flux_map, edge_z, flat_bcs,
                                          e_hat)
        lu_J   = spla.splu(J.tocsc())
        r_diff = enforce_bcs(
            self.residual(u_star, z, geom, rhs_flat, edge_z, e_hat),
            u_star, flat_bcs)
        r_np   = r_diff.detach().cpu().double().numpy()
        corr   = torch.from_numpy(lu_J.solve(r_np))
        if dev.type == 'cuda':
            corr = corr.to(device=dev, dtype=torch.float64)
        u_corrected = u_star - corr
        if not torch.is_grad_enabled():
            return u_corrected
        return u_corrected - SparseSolve.apply(r_diff, lu_J) + corr

    # TODO simplify call signature
    def forward(self, z, geom, bcs, rhs=None, cache=None, z_v=None, z_t=None, z_edge=None):
        problem  = cache or self.prepare(geom, bcs, rhs)
        N        = geom.N
        # enforce_bcs / Picard operate in normalized space: vals / scale
        flat_bcs = [(f * N + nodes, vals / self.field_scale[f])
                    for f, nodes, vals in bcs]
        edge_z   = self.edge_features(geom, z_v, z_t, z_edge)
        # if direct 1 form features are provided, concatenate with the contraction features
        e_hat    = (geom.dx / geom.r.unsqueeze(1)) if (self.edge_dirs or self.vec_fields) else None

        # TODO simplify branched solver path...
        if self.solver_type == 'picard':
            u_flat = self._picard(z, geom, flat_bcs, problem['u0'],
                                  problem['rhs'], edge_z, e_hat)
        else:
            def res(u):
                return enforce_bcs(
                    self.residual(u, z, geom, problem['rhs'], edge_z, e_hat),
                    u, flat_bcs)

            def assemble_J(u):
                return assemble_sparse_jacobian(
                    N, self.n_fields, geom, self.eps_vec, self.gamma,
                    u, z, self.flux_map, edge_z, flat_bcs, e_hat)

            u_flat = self.solver.solve(res, flat_bcs, problem['u0'],
                                       assemble_J=assemble_J)
        # de-normalize: model solved for u_tilde = u / scale
        u_scaled = u_flat.reshape(self.n_fields, N).T   # (N, F) normalized
        return u_scaled * self.field_scale.to(u_scaled.device)  # (N, F) physical
