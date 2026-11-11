"""Numerical experiment for the batched IVP paper: d = 1, s = 2.

The algorithms follow Lemma 4 and Section 6 of the paper with degree-one (bilinear) interpolation,
which is the tensor interpolation of coordinate degree r = 1 used for s = 2.
Every lattice node and every node of a random-time slice is charged as a query,
including slice nodes that the reconstruction does not end up using.
"""
import json
import math
import sys

import numpy as np
from scipy.integrate import solve_ivp


def f(t, x):
    return 0.5 * np.sin(2 * np.pi * t + 3 * x) + 0.4 * np.cos(x - 3 * t)


# Bounds for the test field: |f| <= 0.9, |f_x| <= 1.9, |f_tt| <= 2 pi^2 + 3.6, |f_xx| <= 4.9.
LIP = 1.9
BIL = (2 * np.pi ** 2 + 3.6 + 4.9) / 8.0          # bilinear error <= BIL * h^2
GRON = (math.exp(LIP) - 1.0) / LIP                 # Gronwall factor on [0, 1]
DELTA_CONST = BIL * GRON                           # certified ||y_f - z|| <= DELTA_CONST * h^2


class Oracle:
    def __init__(self):
        self.count = 0

    def charge(self, k):
        self.count += int(k)


class LatticeInterpolant:
    """Bilinear interpolant on t_i = i/nt, x_k = k*hx, using nodes klo[i]..khi[i] in layer i."""

    def __init__(self, nt, hx, klo, khi, oracle):
        self.nt = nt
        self.hx = hx
        self.klo = [int(k) for k in klo]
        self.vals = []
        for i in range(nt + 1):
            ks = np.arange(klo[i], khi[i] + 1)
            self.vals.append(f(i / nt, ks * hx).tolist())
            oracle.charge(len(ks))

    def node(self, i, k):
        j = k - self.klo[i]
        if j < 0 or j >= len(self.vals[i]):
            raise RuntimeError("trajectory left the queried tube")
        return self.vals[i][j]

    def eval(self, t, x):
        i = min(int(t * self.nt), self.nt - 1)
        th = t * self.nt - i
        u = x / self.hx
        k = math.floor(u)
        xi = u - k
        a0, a1 = self.node(i, k), self.node(i, k + 1)
        b0, b1 = self.node(i + 1, k), self.node(i + 1, k + 1)
        lo = a0 + xi * (a1 - a0)
        hi = b0 + xi * (b1 - b0)
        return lo + th * (hi - lo), ((1 - th) * (a1 - a0) + th * (b1 - b0)) / self.hx


def solve_aux(F, nsteps):
    """RK4 for z' = F(t, z) and L' = F_x(t, z), so that P = exp(L)."""
    k = 1.0 / nsteps
    z = L = 0.0
    zs, Ls = [0.0], [0.0]
    for m in range(nsteps):
        t = m * k
        f1, g1 = F.eval(t, z)
        f2, g2 = F.eval(t + k / 2, z + k / 2 * f1)
        f3, g3 = F.eval(t + k / 2, z + k / 2 * f2)
        f4, g4 = F.eval(t + k, z + k * f3)
        z += k * (f1 + 2 * f2 + 2 * f3 + f4) / 6
        L += k * (g1 + 2 * g2 + 2 * g3 + g4) / 6
        zs.append(z)
        Ls.append(L)
    return np.linspace(0.0, 1.0, nsteps + 1), np.array(zs), np.array(Ls)


def slice_interp(u, x, hj, cen, radius):
    """Linear interpolation of f(u, .) at x on the grid hj*Z, plus the window of the slice."""
    kl = np.floor((cen - radius) / hj).astype(np.int64)
    kh = np.ceil((cen + radius) / hj).astype(np.int64)
    k = np.floor(x / hj).astype(np.int64)
    if np.any(k < kl) or np.any(k + 1 > kh):
        raise RuntimeError("trajectory left the random-time slice")
    xi = x / hj - k
    return (1 - xi) * f(u, k * hj) + xi * f(u, (k + 1) * hj), kl, kh


def correction(rng, tg, z, L, F, center, radius, h, levels, M, oracle):
    """Multilevel randomized correction of Lemma 3 with the allocation (eq:allocate)."""
    hs = [h * 2.0 ** (-j) for j in range(levels + 1)]
    cbound = [2 * radius / hj + 4 for hj in hs]
    cbound = [c + (2 * radius / hs[j - 1] + 4 if j else 0) for j, c in enumerate(cbound)]
    a = [hj ** 2 for hj in hs]
    V = sum(aj * math.sqrt(cj) for aj, cj in zip(a, cbound))
    times, weights = [], []
    for j, hj in enumerate(hs):
        N = math.ceil(M * a[j] / (V * math.sqrt(cbound[j])))
        U = rng.random(N)
        zu = np.interp(U, tg, z)
        Lu = np.interp(U, tg, L)
        cen = center(U)
        fine, kl, kh = slice_interp(U, zu, hj, cen, radius)
        nodes = kh - kl + 1
        if j == 0:
            coarse = np.array([F.eval(ui, xi)[0] for ui, xi in zip(U, zu)])
        else:
            coarse, kl2, kh2 = slice_interp(U, zu, hs[j - 1], cen, radius)
            # coarse nodes 2k lying outside the fine window are charged separately
            outside = np.maximum(0, (kl + 1) // 2 - kl2) + np.maximum(0, kh2 - kh // 2)
            nodes = nodes + outside
        oracle.charge(nodes.sum())
        times.append(U)
        weights.append(np.exp(-Lu) * (fine - coarse) / N)
    times = np.concatenate(times)
    weights = np.concatenate(weights)
    order = np.argsort(times)
    cums = np.concatenate([[0.0], np.cumsum(weights[order])])
    H = cums[np.searchsorted(times[order], tg, side="right")]
    return z + np.exp(L) * H


def global_interpolant(M, oracle):
    nt = math.ceil(math.sqrt(M))
    hx = 1.0 / nt
    kmax = math.ceil((1 + 2 * hx) / hx)
    F = LatticeInterpolant(nt, hx, [-kmax] * (nt + 1), [kmax] * (nt + 1), oracle)
    return F, nt, hx


def reference(tg):
    sol = solve_ivp(lambda t, y: f(t, y), (0.0, 1.0), [0.0], method="DOP853",
                    rtol=1e-13, atol=1e-15, dense_output=True)
    return sol.sol(tg)[0]


def one_batch(M, reps, seed):
    det = Oracle()
    F, nt, hx = global_interpolant(M, det)
    nsteps = nt * math.ceil(20000 / nt)
    tg, z, L = solve_aux(F, nsteps)
    yref = reference(tg)
    err_det = np.max(np.abs(z - yref))
    target = M ** (-0.625)                      # h_J ~ M^{-(s+1/2)/(s(d+1))}
    levels = max(0, math.ceil(math.log2(hx / target)))
    rng = np.random.default_rng(seed)
    errs, counts = [], []
    for _ in range(reps):
        orc = Oracle()
        y = correction(rng, tg, z, L, F, lambda u: np.zeros_like(u), 1 + 2 * hx,
                       hx, levels, M, orc)
        errs.append(np.max(np.abs(y - yref)))
        counts.append(det.count + orc.count)
    return dict(M=M, n_det=det.count, err_det=err_det, n=float(np.mean(counts)),
                rms=float(np.sqrt(np.mean(np.square(errs)))), levels=levels)


def two_batch(M, reps, seed):
    orc1 = Oracle()
    F1, nt1, hx1 = global_interpolant(M, orc1)
    tg1, z1, _ = solve_aux(F1, nt1 * math.ceil(20000 / nt1))
    delta = DELTA_CONST * hx1 ** 2               # certified radius after batch 1
    h = 1.0 / M
    ti = np.arange(M + 1) * h
    z1i = np.interp(ti, tg1, z1)
    lo = np.floor((z1i - delta - 3 * h) / h).astype(np.int64)
    hi = np.ceil((z1i + delta + 3 * h) / h).astype(np.int64)
    klo = np.minimum(lo, np.concatenate([[lo[0]], lo[:-1]]))
    khi = np.maximum(hi, np.concatenate([[hi[0]], hi[:-1]]))
    orc2 = Oracle()
    F2 = LatticeInterpolant(M, h, klo, khi, orc2)
    tg, z, L = solve_aux(F2, 64 * M)
    yref = reference(tg)
    err_det = np.max(np.abs(z - yref))
    radius = delta + 2 * h
    target = M ** (-1.25)                       # h_J^s <= M^{-s-1/2}
    levels = max(0, math.ceil(math.log2(h / target)))
    rng = np.random.default_rng(seed)
    z1fun = lambda u: np.interp(u, tg1, z1)
    errs, counts = [], []
    for _ in range(reps):
        orc = Oracle()
        y = correction(rng, tg, z, L, F2, z1fun, radius, h, levels, M, orc)
        errs.append(np.max(np.abs(y - yref)))
        counts.append(orc1.count + orc2.count + orc.count)
    return dict(M=M, n_det=orc1.count + orc2.count, err_det=err_det,
                n=float(np.mean(counts)), rms=float(np.sqrt(np.mean(np.square(errs)))),
                levels=levels, delta=delta,
                batch1_err=float(np.max(np.abs(np.interp(tg, tg1, z1) - yref))))


def slope(rows, nkey, ekey, last=4):
    x = np.log([r[nkey] for r in rows[-last:]])
    y = np.log([r[ekey] for r in rows[-last:]])
    return float(np.polyfit(x, y, 1)[0])


if __name__ == "__main__":
    reps = 50
    one = [one_batch(M, reps, 1000 + i) for i, M in
           enumerate([100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000])]
    two = [two_batch(M, reps, 2000 + i) for i, M in
           enumerate([25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800])]
    for r in two:
        assert r["batch1_err"] <= r["delta"], "certified radius violated"
    out = dict(one=one, two=two,
               slopes=dict(one_det=slope(one, "n_det", "err_det"),
                           one_rand=slope(one, "n", "rms"),
                           two_det=slope(two, "n_det", "err_det"),
                           two_rand=slope(two, "n", "rms")))
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


