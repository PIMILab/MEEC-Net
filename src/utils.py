import torch
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def tsolve(A, b):
    """Direct sparse solve A x = b (no gradients — used only in DC-PSE assembly)."""
    A_sp = torch_sparse_to_scipy_csr(A.coalesce())
    x_np = spla.spsolve(A_sp, b.detach().cpu().numpy())
    return torch.from_numpy(x_np).to(device=b.device, dtype=b.dtype)


def tenforce(A, b, D=None, x=None, *, diag=1.0):
    if D is None:
        return A, b

    D = D.to(device=b.device, dtype=torch.long)

    if D.numel() == 0:
        return A, b

    A = A.coalesce()
    n = A.shape[0]

    rows = A.indices()[0]
    cols = A.indices()[1]
    vals = A.values()

    # row mask: keep entries NOT in D
    is_dirichlet_row = torch.zeros(n, dtype=torch.bool, device=b.device)
    is_dirichlet_row[D] = True
    keep = ~is_dirichlet_row[rows]

    kept_idx = torch.stack([rows[keep], cols[keep]], dim=0)
    kept_vals = vals[keep]

    # add diagonal entries for D
    diag_idx = torch.stack([D, D], dim=0)
    diag_vals = torch.full((D.numel(),), float(diag), dtype=vals.dtype, device=vals.device)

    new_idx = torch.cat([kept_idx, diag_idx], dim=1)
    new_vals = torch.cat([kept_vals, diag_vals], dim=0)

    A2 = torch.sparse_coo_tensor(new_idx, new_vals, size=A.shape, device=A.device, dtype=vals.dtype).coalesce()

    # RHS
    b2 = b.clone()
    if x is None:
        x_t = torch.zeros((D.numel(),) + b2.shape[1:], dtype=b2.dtype, device=b2.device)
    else:
        x_t = torch.as_tensor(x, dtype=b2.dtype, device=b2.device)
        if x_t.ndim == 0:
            x_t = x_t.expand((D.numel(),) + b2.shape[1:])
        elif x_t.shape[0] != D.numel():
            # allow x shaped like b (n, ...) and index into it
            if x_t.shape == b2.shape:
                x_t = x_t[D]
            else:
                raise ValueError("x must be scalar, shape (len(D), ...), or same shape as b")

    if b2.ndim == 1:
        b2[D] = float(diag) * x_t.reshape(-1)
    else:
        b2[D, ...] = float(diag) * x_t

    return A2, b2


def torch_sparse_to_scipy_csr(A_torch) -> sp.csr_matrix:
    A = A_torch.coalesce()
    idx = A.indices().cpu().numpy()
    val = A.values().detach().cpu().numpy()
    return sp.csr_matrix((val, (idx[0], idx[1])), shape=A.shape)


def eps_ball_graph(pts, eps, chunk=2048):
    """All pairs (i,j) with i<j and ||pts[i]-pts[j]|| < eps.  Pure PyTorch.

    Returns edges (n_e, 2) long tensor.  Works with pts.requires_grad.
    Chunked to keep memory O(chunk * N) instead of O(N^2).
    """
    N = pts.shape[0]
    pairs = []
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        # distances from pts[lo:hi] to pts[hi:] (upper triangle only)
        d = torch.cdist(pts[lo:hi], pts[hi:])        # (hi-lo, N-hi)
        ii, jj = (d < eps).nonzero(as_tuple=True)
        pairs.append(torch.stack([ii + lo, jj + hi], dim=1))
        # within-chunk upper triangle
        if hi - lo > 1:
            dc = torch.cdist(pts[lo:hi], pts[lo:hi])  # (chunk, chunk)
            mask = torch.triu(dc < eps, diagonal=1)
            ii, jj = mask.nonzero(as_tuple=True)
            pairs.append(torch.stack([ii + lo, jj + lo], dim=1))
    return torch.cat(pairs, dim=0) if pairs else torch.zeros(0, 2, dtype=torch.long)


def mean_nn_distance(pts, k=2, chunk=2048):
    """Mean nearest-neighbor distance. Pure PyTorch, no scipy."""
    N = pts.shape[0]
    nn_dist = torch.full((N,), float("inf"), dtype=pts.dtype, device=pts.device)
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        d = torch.cdist(pts[lo:hi], pts)  # (chunk, N)
        d[torch.arange(hi - lo), torch.arange(lo, hi)] = float("inf")  # exclude self
        nn_dist[lo:hi] = d.min(dim=1).values
    return nn_dist.mean()


def max_nn_distance(pts, chunk=2048):
    """Maximum nearest-neighbor distance. Ensures every point has a neighbor."""
    N = pts.shape[0]
    nn_dist = torch.full((N,), float("inf"), dtype=pts.dtype, device=pts.device)
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        d = torch.cdist(pts[lo:hi], pts)  # (chunk, N)
        d[torch.arange(hi - lo), torch.arange(lo, hi)] = float("inf")
        nn_dist[lo:hi] = d.min(dim=1).values
    return nn_dist.max()


def boundary_geometry(pts, bd_mask):
    """Outward unit normals and Voronoi arc lengths at boundary nodes (2D only).

    Sorts boundary nodes CCW by angle from centroid, estimates each node's
    tangent by central difference on the closed loop.  From the tangent vector t:
      - unit normal  = 90°-CW rotation of t/|t|  (outward for a CCW curve)
      - arc length   = |t| / 2  (Voronoi measure: mean of two adjacent segment lengths)

    Parameters
    ----------
    pts     : (N, 2) point cloud
    bd_mask : (N,) bool, True for boundary nodes

    Returns
    -------
    normals     : (N, 2)  outward unit normals  (0 at interior nodes)
    arc_lengths : (N,)    attributed arc length  (0 at interior nodes)
    """
    idx = bd_mask.nonzero().squeeze(1)
    bp = pts[idx].double()
    cx, cy = bp[:, 0].mean(), bp[:, 1].mean()
    angles = torch.atan2(bp[:, 1] - cy, bp[:, 0] - cx)
    order = angles.argsort()
    inv_order = torch.argsort(order)
    bp_s = bp[order]
    t = bp_s.roll(-1, 0) - bp_s.roll(1, 0)          # central-difference tangent
    t_len = t.norm(dim=1).clamp(min=1e-14)
    n = torch.stack([t[:, 1], -t[:, 0]], dim=1) / t_len.unsqueeze(1)  # CW → outward
    s = t_len / 2
    normals     = torch.zeros(len(pts), 2, dtype=torch.float64)
    arc_lengths = torch.zeros(len(pts),    dtype=torch.float64)
    normals[idx]     = n[inv_order]
    arc_lengths[idx] = s[inv_order]
    return normals, arc_lengths


def boundary_geometry_3d(pts, bd_mask, eps):
    """Outward unit normals at 3D boundary nodes via local PCA.

    For each boundary node: collect boundary neighbours within eps, compute
    covariance of displacements, take the eigenvector of the smallest
    eigenvalue as the normal, orient outward via the cloud centroid.

    Parameters
    ----------
    pts     : (N, 3) point cloud
    bd_mask : (N,) bool
    eps     : neighbourhood radius

    Returns
    -------
    normals : (N, 3)  outward unit normals (zero at interior nodes)
    """
    bd_idx   = bd_mask.nonzero().squeeze(1)
    centroid = pts.mean(0)
    normals  = torch.zeros(len(pts), 3, dtype=torch.float64)
    for j in bd_idx.tolist():
        p = pts[j]
        nbr = bd_mask & ((pts - p).norm(dim=1) < eps) & ((pts - p).norm(dim=1) > 0)
        nbrs = pts[nbr]
        if len(nbrs) < 2:
            n = p - centroid
        else:
            dX = nbrs - p
            _, vecs = torch.linalg.eigh((dX.T @ dX) / len(dX))
            n = vecs[:, 0]                           # smallest eigenvector
        if (n @ (p - centroid)) < 0:
            n = -n
        normals[j] = n / n.norm().clamp(min=1e-14)
    return normals


def robin_apply(L, rhs, dec, g, alpha, beta, normals, K=2):
    """Replace Robin rows: alpha*p + beta*dp/dn = g (degree-1 WLS, beta != 0)."""
    L = L.coalesce()
    ei, ej = dec.edges.unbind(1)

    neu_flag = torch.zeros(dec.N, dtype=torch.bool)
    neu_flag[dec.neumann_idx] = True
    keep = ~neu_flag[L.indices()[0]]
    rows = L.indices()[0][keep].tolist()
    cols = L.indices()[1][keep].tolist()
    vals = L.values()[keep].tolist()
    rhs_out = rhs.clone()

    for qi, j in enumerate(dec.neumann_idx.tolist()):
        nbrs = torch.cat([ej[ei == j], ei[ej == j]])
        if len(nbrs) == 0:
            continue
        dx  = dec.pts[nbrs] - dec.pts[j]
        w   = (1.0 - dx.norm(dim=1) / dec.eps).clamp(min=0).pow(K)
        c   = w * (dx @ torch.linalg.solve((dx * w[:, None]).T @ dx, normals[j].double()))
        vol = dec.M0_diag[j].item()
        rows.append(j); cols.append(j); vals.append((alpha - beta * c.sum().item()) * vol)
        for ki, k in enumerate(nbrs.tolist()):
            rows.append(j); cols.append(k); vals.append(beta * c[ki].item() * vol)
        rhs_out[j] = float(g[qi]) * vol

    return torch.sparse_coo_tensor(
        torch.tensor([rows, cols], dtype=torch.long),
        torch.tensor(vals, dtype=torch.float64),
        size=(dec.N, dec.N),
    ).coalesce(), rhs_out


def neumann_apply(L, rhs, dec, q, normals, K=2):
    """Neumann BC: robin_apply with alpha=0, beta=1."""
    return robin_apply(L, rhs, dec, q, 0.0, 1.0, normals, K=K)


def neumann_ghost_points(pts, bd_mask, neu_mask, normals=None, h_scale=None):
    """Augment cloud with one interior ghost per Neumann node, placed h inward along n_hat."""
    if normals is None:
        normals, _ = boundary_geometry(pts, bd_mask)
    if h_scale is None:
        h_scale = float(mean_nn_distance(pts))
    idx    = neu_mask.nonzero().squeeze(1)
    ghosts = pts[idx] - h_scale * normals[idx]
    zeros  = torch.zeros(len(ghosts), dtype=torch.bool)
    return torch.cat([pts, ghosts]), torch.cat([bd_mask, zeros]), torch.cat([neu_mask, zeros])
