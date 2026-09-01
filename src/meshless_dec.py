"""Meshfree DEC on 2D and 3D point clouds (PyTorch).

L = d0^T diag(M1) d0  ≈  -M0 nabla^2

M1 assembly methods (set via MeshlessDEC.assemble(method=...)):

  'rkpm'       — RKPM-normalised Wendland kernel.  O(N), always positive,
                 fully differentiable through pts.  Use for ML / shape optimisation.

  'dc_pse'     — DC-PSE (Schrader et al. 2010).  O(h²) consistency.
                 Not differentiable through pts.

  'dc_pse_pos' — DC-PSE + m[e] >= 0 via OSQP (requires osqp package).
                 Eliminates negative M1 entries near asymmetric stencils.

Dimension inferred from pts.shape[1] (2D or 3D).
"""

import warnings
import torch
from utils import (tsolve, tenforce, eps_ball_graph, max_nn_distance,
                   boundary_geometry, boundary_geometry_3d,
                   neumann_apply as _neumann_apply,
                   robin_apply as _robin_apply)

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")


def sparse_diag(v):
    n = len(v)
    idx = torch.arange(n).unsqueeze(0).expand(2, -1)
    return torch.sparse_coo_tensor(idx, v, (n, n)).coalesce()


def choose_epsilon(pts, C=2.0):
    """Pick support radius eps = C * h_max, where h_max is the max nearest-neighbor distance.

    Using the maximum ensures every point has at least one neighbor within eps.
    """
    h_max = float(max_nn_distance(pts.detach()))
    return C * h_max


def choose_epsilon_knn(pts, K=8):
    """Pick the smallest eps such that every node has at least K neighbours within eps.

    Concretely: eps = max over all nodes of their K-th nearest-neighbour distance.
    This is strictly better than C * h_max because it guarantees a minimum
    stencil size everywhere, regardless of local density variations.

    Parameters
    ----------
    pts : (N, d) float tensor
    K   : minimum number of neighbours required per node (default 8, good for 2D DC-PSE;
          use 15-20 for 3D)
    """
    from scipy.spatial import KDTree
    pts_np = pts.detach().cpu().numpy()
    tree   = KDTree(pts_np)
    # query K+1 because the point itself is included as distance 0
    dists, _ = tree.query(pts_np, k=min(K + 1, len(pts_np)))
    kth_dist = dists[:, -1]   # K-th neighbour distance for every node
    return float(kth_dist.max())


class MeshlessDEC:

    def __init__(self, pts, eps, boundary_mask, neumann_mask=None):
        self.pts = pts.double()
        self.eps = float(eps)
        self.N   = len(pts)
        self.dim = pts.shape[1]          # 2 or 3
        self.boundary_mask = boundary_mask.bool()
        self.neumann_mask = neumann_mask.bool() if neumann_mask is not None \
            else torch.zeros(self.N, dtype=torch.bool)
        self.dirichlet_mask = self.boundary_mask & ~self.neumann_mask

        # Geometric classification
        self.interior_mask = ~self.boundary_mask
        self.interior_idx = self.interior_mask.nonzero().squeeze(1)
        self.boundary_idx = self.boundary_mask.nonzero().squeeze(1)
        self.dirichlet_idx = self.dirichlet_mask.nonzero().squeeze(1)
        self.neumann_idx = self.neumann_mask.nonzero().squeeze(1)

        # Assembly classification: "free" = interior ∪ Neumann (unknown DOFs)
        #   M0 volumes: computed for free nodes (Neumann not special-cased)
        #   M1 moments: interior nodes only (boundary half-stencils break QP)
        self.free_mask = ~self.dirichlet_mask
        self.free_idx = self.free_mask.nonzero().squeeze(1)

    # ── graph ────────────────────────────────────────────────

    def build_graph(self):
        """eps-ball graph.  Sets edges (n_e, 2), dx, r.

        Uses pure PyTorch so dx and r are differentiable w.r.t. pts.
        """
        self.edges = eps_ball_graph(self.pts.detach(), self.eps)
        self.n_edges = len(self.edges)
        self.dx = self.pts[self.edges[:, 1]] - self.pts[self.edges[:, 0]]
        self.r = self.dx.norm(dim=1)

    # ── d0: node-to-edge coboundary ──────────────────────────

    def assemble_d0(self):
        """(d0 u)_e = u_j - u_i.  Sparse (n_edges, N)."""
        e = torch.arange(self.n_edges)
        ei, ej = self.edges.unbind(1)
        self.d0 = torch.sparse_coo_tensor(
            torch.stack([torch.cat([e, e]), torch.cat([ei, ej])]),
            torch.cat([-torch.ones(self.n_edges, dtype=torch.float64),
                        torch.ones(self.n_edges, dtype=torch.float64)]),
            (self.n_edges, self.N)).coalesce()

    # ── M0: node volumes ────────────────────────────────────

    def assemble_M0(self, domain_area=1.0):
        """Shepard volumes for free nodes (interior ∪ Neumann), normalised to domain_area."""
        ei, ej = self.edges.unbind(1)
        w = (1 - self.r / self.eps).clamp(min=0).pow(4)
        kappa = torch.ones(self.N, dtype=torch.float64)   # self-contribution: kappa(0) = 1
        kappa.scatter_add_(0, ei, w)
        kappa.scatter_add_(0, ej, w)

        ff = self.free_idx
        rho = 1.0 / kappa[ff].clamp(min=1e-30)
        self.M0_diag = torch.zeros(self.N, dtype=torch.float64)
        self.M0_diag[ff] = rho * (domain_area / rho.sum())
        self.M0 = sparse_diag(self.M0_diag)

    # ── M1: edge Hodge star ─────────────────────────────────

    def _assemble_M1_from_kernel(self, phi):
        """Shared RKPM normalisation for any non-negative edge kernel phi (E,).

        Standard RKPM trace condition:
            sum_{e ni i} M1[e] * r_e^2  =  trace_scale * M0[i]

        Per-node coefficient:
            c[i] = trace_scale * M0[i] / sum_{e ni i} phi(r_e) * r_e^2

        Per-edge weight:
            M1[e] = phi[e] * coeff

        where coeff uses 0.5*(c_i + c_j) for free-free edges and the free-node
        coefficient alone when one endpoint is Dirichlet (avoids halving the
        trace contribution of boundary-adjacent edges).
        """
        ei, ej = self.edges.unbind(1)

        phir2_sum = phi.new_zeros(self.N)
        phir2_sum.scatter_add_(0, ei, phi * self.r.pow(2))
        phir2_sum.scatter_add_(0, ej, phi * self.r.pow(2))

        trace_scale = 2.0 * self.dim          # 4 in 2D, 6 in 3D
        c = trace_scale * self.M0_diag / phir2_sum.clamp(min=1e-28)  # 0 on Dirichlet

        ci, cj = c[ei], c[ej]
        free_i, free_j = self.free_mask[ei], self.free_mask[ej]
        coeff = torch.where(
            free_i & free_j, 0.5 * (ci + cj),
            torch.where(free_i, ci, cj)
        )

        self.M1_diag = phi * coeff
        self.M1      = sparse_diag(self.M1_diag)

    def assemble_M1_rkpm(self, K=2):
        """RKPM-normalised Wendland kernel.

        Kernel:  phi(r) = (1 - r/eps)^K_+
        Normalisation: c[i] = trace_scale * M0[i] / sum_{e ni i} phi(r_e) * r_e^2
        M1[e] = phi(r_e) * coeff  (see _assemble_M1_from_kernel for coeff rule)

        O(N) cost, always positive, differentiable through pts via autograd.
        """
        phi = (1 - self.r / self.eps).clamp(min=0).pow(K)
        self._assemble_M1_from_kernel(phi)

    def assemble_M1_dc_pse(self, K=2):
        """Edge Hodge star via DC-PSE min-norm QP on interior nodes.

        2D: 5 moment conditions, trace RHS = 4 * M0.
        3D: 9 moment conditions, trace RHS = 6 * M0.
        Boundary nodes excluded; use neumann_apply() after assembling L.
        Not differentiable through pts.
        """
        ei, ej = self.edges.unbind(1)
        # Nondimensionalize by eps to keep all moment columns O(1).
        dx = self.dx / self.eps
        phi = (1 - self.r / self.eps).clamp(min=0).pow(K) + 1e-14

        # ── dimension-specific moment basis (nondimensional coordinates) ──
        if self.dim == 2:
            x, y = dx[:, 0], dx[:, 1]
            mom = torch.stack([
                x, y, x*y, x**2 - y**2, x**2 + y**2], dim=1)          # (E, 5)
            mom_neg = torch.stack([
                -x, -y, x*y, x**2 - y**2, x**2 + y**2], dim=1)
            trace_scale = 4.0
        else:
            x, y, z = dx[:, 0], dx[:, 1], dx[:, 2]
            mom = torch.stack([
                x, y, z, x*y, x*z, y*z,
                x**2 - y**2, y**2 - z**2, x**2 + y**2 + z**2], dim=1)  # (E, 9)
            mom_neg = torch.stack([
                -x, -y, -z, x*y, x*z, y*z,
                x**2 - y**2, y**2 - z**2, x**2 + y**2 + z**2], dim=1)
            trace_scale = 6.0

        M  = mom.shape[1]                # 5 or 9
        ii = self.interior_idx
        N_I = len(ii)
        imap = -torch.ones(self.N, dtype=torch.long)
        imap[ii] = torch.arange(N_I)
        loc_i = imap[ei]
        loc_j = imap[ej]
        m_i   = loc_i >= 0
        m_j   = loc_j >= 0
        m_both = m_i & m_j
        boff  = torch.arange(M, dtype=torch.long) * N_I

        # ── S = B diag(phi) B^T — vectorized over 5×5 blocks ──
        def _outer_block(mask, loc_r, loc_c, mom_r, mom_c):
            """Batched rank-1 COO contributions for all M×M blocks at once."""
            n = mask.sum()
            if n == 0:
                z = torch.zeros(0, dtype=torch.long)
                return z, z, torch.zeros(0, dtype=torch.float64)
            lr = loc_r[mask]; lc = loc_c[mask]
            mr = mom_r[mask]; mc = mom_c[mask]
            p  = phi[mask]
            rows = boff[:, None] + lr[None, :]                          # (M, n)
            cols = boff[:, None] + lc[None, :]
            vals = p[None, None, :] * mr.T[:, None, :] * mc.T[None, :, :]  # (M, M, n)
            rows = rows[:, None, :].expand(M, M, n)
            cols = cols[None, :, :].expand(M, M, n)
            return rows.reshape(-1), cols.reshape(-1), vals.reshape(-1)

        # four contributions: ii, jj, ij, ji
        r1, c1, v1 = _outer_block(m_i,    loc_i, loc_i, mom,     mom)
        r2, c2, v2 = _outer_block(m_j,    loc_j, loc_j, mom_neg, mom_neg)
        r3, c3, v3 = _outer_block(m_both, loc_i, loc_j, mom,     mom_neg)
        r4, c4, v4 = _outer_block(m_both, loc_j, loc_i, mom_neg, mom)

        n_S = M * N_I
        S = torch.sparse_coo_tensor(
            torch.stack([torch.cat([r1,r2,r3,r4]),
                         torch.cat([c1,c2,c3,c4])]),
            torch.cat([v1,v2,v3,v4]), (n_S, n_S)).coalesce()

        rhs = torch.zeros(n_S, dtype=torch.float64)
        # trace channel is a degree-2 moment; rescale to match the /eps^2
        # nondimensionalization of the moment basis above.
        rhs[(M-1) * N_I:] = trace_scale * self.M0_diag[ii] / self.eps**2

        lam = tsolve(S, rhs)


        # ── B^T @ lam — vectorized over 5 moment channels ──
        def _B_block(mask, loc, mom_blk):
            n = mask.sum()
            if n == 0:
                z = torch.zeros(0, dtype=torch.long)
                return z, z, torch.zeros(0, dtype=torch.float64)
            e_idx = mask.nonzero().squeeze(1)       # (n,)  edge indices
            l     = loc[e_idx]                      # (n,)  local node index
            m     = mom_blk[e_idx]                  # (n, M)
            rows  = boff[:, None] + l[None, :]      # (M, n)
            cols  = e_idx[None, :].expand(M, n)     # (M, n)
            return rows.reshape(-1), cols.reshape(-1), m.T.reshape(-1)

        br1, bc1, bv1 = _B_block(m_i, loc_i, mom)
        br2, bc2, bv2 = _B_block(m_j, loc_j, mom_neg)

        B = torch.sparse_coo_tensor(
            torch.stack([torch.cat([br1, br2]), torch.cat([bc1, bc2])]),
            torch.cat([bv1, bv2]), (n_S, self.n_edges)).coalesce()

        Bt_lam = torch.sparse.mm(B.t().coalesce(), lam.unsqueeze(1)).squeeze(1)
        self.M1_diag = phi * Bt_lam
        self.M1 = sparse_diag(self.M1_diag)

    def assemble_M1_dc_pse_pos(self, K=2):
        """DC-PSE edge Hodge star with non-negativity constraint (requires osqp).

        Solves the same moment-constrained QP as assemble_M1_dc_pse but adds
        m[e] >= 0 for all edges:

            min  (1/2) m^T diag(1/phi) m
            s.t. B m = b   (DC-PSE moment conditions)
                 m >= 0

        The equality-only dual is replaced by an OSQP primal solve.  Solution
        is polished by OSQP so accuracy matches the unconstrained version on
        well-conditioned stencils and is strictly non-negative everywhere else.
        """
        try:
            import osqp
        except ImportError:
            raise ImportError(
                "'dc_pse_pos' requires osqp.  Install with: pip install osqp"
            )
        import scipy.sparse as sp_sci
        import numpy as np

        ei, ej = self.edges.unbind(1)
        # Nondimensionalize by eps — see assemble_M1_dc_pse for why (keeps
        # degree-1/degree-2 moment columns at the same O(1) scale).
        dx  = self.dx / self.eps
        phi = (1 - self.r / self.eps).clamp(min=0).pow(K) + 1e-14
        E   = self.n_edges

        # ── moment basis (same as dc_pse) ───────────────────────────────
        if self.dim == 2:
            x, y = dx[:, 0], dx[:, 1]
            mom     = torch.stack([x, y, x*y, x**2 - y**2, x**2 + y**2], dim=1)
            mom_neg = torch.stack([-x, -y, x*y, x**2 - y**2, x**2 + y**2], dim=1)
            trace_scale = 4.0
        else:
            x, y, z = dx[:, 0], dx[:, 1], dx[:, 2]
            mom     = torch.stack([x, y, z, x*y, x*z, y*z,
                                   x**2 - y**2, y**2 - z**2,
                                   x**2 + y**2 + z**2], dim=1)
            mom_neg = torch.stack([-x, -y, -z, x*y, x*z, y*z,
                                   x**2 - y**2, y**2 - z**2,
                                   x**2 + y**2 + z**2], dim=1)
            trace_scale = 6.0

        M    = mom.shape[1]
        ii   = self.interior_idx
        N_I  = len(ii)
        imap = -torch.ones(self.N, dtype=torch.long)
        imap[ii] = torch.arange(N_I)
        loc_i = imap[ei];  loc_j = imap[ej]
        m_i   = loc_i >= 0;  m_j = loc_j >= 0
        boff  = torch.arange(M, dtype=torch.long) * N_I

        # ── build B: (M*N_I, E) in COO then CSC ─────────────────────────
        def _B_block_np(mask, loc, mom_blk):
            n = int(mask.sum())
            if n == 0:
                return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0)
            e_idx = mask.nonzero().squeeze(1)
            l     = loc[e_idx]
            m     = mom_blk[e_idx]               # (n, M)
            rows  = (boff[:, None] + l[None, :]).reshape(-1).numpy().astype(np.int32)
            cols  = e_idx[None, :].expand(M, n).reshape(-1).numpy().astype(np.int32)
            vals  = m.T.reshape(-1).numpy()
            return rows, cols, vals

        br1, bc1, bv1 = _B_block_np(m_i, loc_i, mom)
        br2, bc2, bv2 = _B_block_np(m_j, loc_j, mom_neg)
        rows_B = np.concatenate([br1, br2])
        cols_B = np.concatenate([bc1, bc2])
        vals_B = np.concatenate([bv1, bv2])

        n_eq = M * N_I
        B_sp = sp_sci.csc_matrix((vals_B, (rows_B, cols_B)), shape=(n_eq, E))

        b = np.zeros(n_eq)
        # trace channel is a degree-2 moment; rescale to match the /eps^2
        # nondimensionalization of the moment basis above.
        b[(M - 1) * N_I:] = (trace_scale * self.M0_diag[ii] / self.eps**2).numpy()

        # ── OSQP: min (1/2) m^T P m  s.t.  B m = b,  m >= 0 ────────────
        # Constraint matrix A = [B; I],  l = [b; 0],  u = [b; +inf]
        phi_np = phi.numpy()
        P = sp_sci.diags(1.0 / phi_np.clip(1e-14), format='csc')
        A = sp_sci.vstack([B_sp, sp_sci.eye(E, format='csc')], format='csc')
        l = np.concatenate([b, np.zeros(E)])
        u = np.concatenate([b, np.full(E, np.inf)])

        solver = osqp.OSQP()
        solver.setup(P, np.zeros(E), A, l, u,
                     warm_starting=True, verbose=False,
                     eps_abs=1e-8, eps_rel=1e-8,
                     max_iter=10_000, polish=True)
        result = solver.solve()

        if result.info.status not in ('solved', 'solved_inaccurate'):
            warnings.warn(
                f"dc_pse_pos OSQP did not converge: {result.info.status}. "
                "Falling back to clamped dc_pse solution."
            )
            m_sol = torch.from_numpy(result.x).clamp(min=0.0)
        else:
            m_sol = torch.from_numpy(result.x).clamp(min=0.0)

        self.M1_diag = m_sol
        self.M1      = sparse_diag(self.M1_diag)

    # ── convenience ──────────────────────────────────────────

    def assemble(self, domain_vol=1.0, K=2, method="dc_pse"):
        """Build graph + d0 + M0 + M1.

        method : {'rkpm', 'dc_pse', 'dc_pse_pos'}
            'rkpm'       — fast, differentiable, O(N), always positive
            'dc_pse'     — accurate, O(h²), not differentiable through pts
            'dc_pse_pos' — dc_pse + m[e]>=0 constraint via OSQP (requires osqp)
        """
        self.build_graph()
        self.assemble_d0()
        self.assemble_M0(domain_vol)
        if method == "rkpm":
            self.assemble_M1_rkpm(K)
        elif method == "dc_pse":
            self.assemble_M1_dc_pse(K)
        elif method == "dc_pse_pos":
            self.assemble_M1_dc_pse_pos(K)
        else:
            raise ValueError(
                f"Unknown M1 method '{method}'. Choose 'rkpm', 'dc_pse', or 'dc_pse_pos'."
            )
        self.m1_method = method

    def _auto_normals(self):
        """Compute outward normals automatically from geometry (2D CCW or 3D PCA)."""
        if self.dim == 2:
            normals, _ = boundary_geometry(self.pts, self.boundary_mask)
        else:
            normals = boundary_geometry_3d(self.pts, self.boundary_mask, self.eps)
        return normals

    def neumann_apply(self, L, rhs, q, normals=None, K=2):
        """Degree-1 WLS Neumann row replacement; normals auto-computed if None."""
        if normals is None:
            normals = self._auto_normals()
        return _neumann_apply(L, rhs, self, q, normals, K=K)

    def robin_apply(self, L, rhs, g, alpha, beta, normals=None, K=2):
        """Degree-1 WLS Robin row replacement (alpha*p + beta*dp/dn = g); normals auto-computed if None."""
        if normals is None:
            normals = self._auto_normals()
        return _robin_apply(L, rhs, self, g, alpha, beta, normals, K=K)

    def to(self, device):
        """Move all assembled tensors to device (in-place). Returns self."""
        for attr in ('pts', 'edges', 'dx', 'r', 'd0', 'M0', 'M0_diag',
                      'M1', 'M1_diag', 'boundary_mask', 'interior_mask',
                      'free_mask', 'dirichlet_mask', 'neumann_mask',
                      'boundary_idx', 'interior_idx', 'dirichlet_idx',
                      'neumann_idx', 'free_idx'):
            v = getattr(self, attr, None)
            if v is not None and isinstance(v, torch.Tensor):
                setattr(self, attr, v.to(device))
        return self

