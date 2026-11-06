"""
Koopman operators, approximated on a dictionary of observables.

The Koopman generator of x' = f(x) acts on an observable psi as the Lie
derivative, (L psi)(x) = grad(psi) . f = L_f psi, so lie_lib already computes
it exactly at any point. gedmd() uses that directly (generator EDMD, Klus et
al. 2020): no time stepping, no finite differences. edmd() is the classical
snapshot version (Williams, Kevrekidis & Rowley 2015) for when only data
pairs (x, F(x)) are available.

Both fit a matrix M on the dictionary, with samples in rows:

    Psi(X) M ~ (L psi)(X)        generator
    Psi(X) K ~ Psi(Y)            flow map, Y = F(X)

An eigenvector xi of M gives an eigenfunction phi(x) = psi(x) . xi, with
L_f phi = lambda phi (generator) or phi(F(x)) = mu phi(x) (flow map).

@author: virati
"""

from itertools import combinations_with_replacement

import autograd.numpy as np
import numpy as onp

from autolie.lie_lib import L_d
from autolie.op_plus import operable


def polynomial_dictionary(n, degree, constant=True):
    """Monomials in n variables up to `degree`, as one vector-valued operable.

    Returns (psi, labels); labels[i] names psi's i-th component, e.g. 'x0^2*x2'.
    """
    terms = [()] if constant else []
    for d in range(1, degree + 1):
        terms += list(combinations_with_replacement(range(n), d))

    def label(t):
        if not t:
            return "1"
        return "*".join(f"x{i}" + (f"^{t.count(i)}" if t.count(i) > 1 else "") for i in sorted(set(t)))

    def psi(x, args=None):
        return np.array([np.prod(np.array([x[i] for i in t])) if t else 1.0 + 0.0 * x[0] for t in terms])

    return operable(psi), [label(t) for t in terms]


class KoopmanApprox:
    """A fitted matrix on a dictionary, with its spectrum, eigenfunctions and modes.

    `kind` is "generator" (continuous time, from gedmd) or "flow" (one step of a
    flow map, from edmd); it decides what predict() does with the eigenvalues.
    """

    def __init__(self, psi, M, residual, kind, Psi, X):
        self.psi = psi
        self.matrix = M
        self.residual = residual
        self.kind = kind
        self.eigenvalues, self.eigenvectors = onp.linalg.eig(M)
        # Koopman modes: the full state written in eigenfunctions. With the
        # state fitted on the dictionary, x ~ psi(x) C, and psi = phi Xi^-1,
        # so x ~ sum_i phi_i(x) V_i with V = Xi^-1 C.
        C, *_ = onp.linalg.lstsq(Psi, X, rcond=None)
        self.state_residual = onp.linalg.norm(Psi @ C - X) / max(onp.linalg.norm(X), 1e-300)
        self.modes = onp.linalg.solve(self.eigenvectors, C.astype(complex))

    def eigenfunction(self, i):
        xi = self.eigenvectors[:, i]
        return lambda x, args=None: onp.dot(onp.asarray(self.psi(x, args)), xi)

    def eigenfunctions(self, x, args=None):
        """All eigenfunctions at x, as a vector phi(x) = psi(x) Xi."""
        return onp.asarray(self.psi(x, args), dtype=float) @ self.eigenvectors

    def predict(self, x0, t, args=None):
        """The state at time(s) t from x0, by Koopman mode decomposition.

        x(t) = sum_i phi_i(x0) exp(lambda_i t) V_i for a generator, with t in
        the system's time units; for a flow map t counts steps and
        exp(lambda_i t) becomes mu_i^t. Exact when the dictionary is
        Koopman-invariant and contains the state (both residuals ~ 0).
        """
        phi0 = self.eigenfunctions(x0, args)
        ts = onp.atleast_1d(onp.asarray(t, dtype=float))
        if self.kind == "generator":
            growth = onp.exp(onp.outer(ts, self.eigenvalues))
        else:
            growth = onp.power.outer(self.eigenvalues.astype(complex), ts).T
        out = ((growth * phi0) @ self.modes).real
        return out[0] if onp.ndim(t) == 0 else out


def _evaluate(fn, X, args=None):
    return onp.array([onp.asarray(fn(x, args), dtype=float) for x in X])


def _fit(Psi, target, psi, kind, X):
    M, *_ = onp.linalg.lstsq(Psi, target, rcond=None)
    resid = onp.linalg.norm(Psi @ M - target) / max(onp.linalg.norm(target), 1e-300)
    return KoopmanApprox(psi, M, resid, kind, Psi, onp.asarray(X, dtype=float))


def gedmd(f, psi, X, args=None):
    """Generator EDMD: fit L_f psi ~ psi M on the sample points X (N x n).

    The derivative of the dictionary along f is computed exactly with L_d,
    so the only approximation is the projection onto span(psi). A relative
    residual near zero means the dictionary is (nearly) Koopman-invariant.
    """
    return _fit(_evaluate(psi, X, args), _evaluate(L_d(psi, f), X, args), psi, "generator", X)


def edmd(psi, X, Y, args=None):
    """Snapshot EDMD: fit psi(Y) ~ psi(X) K from pairs Y[i] = F(X[i])."""
    return _fit(_evaluate(psi, X, args), _evaluate(psi, Y, args), psi, "flow", X)


class BilinearLift:
    """A controlled system x' = f(x) + sum_j g_j(x) u_j, lifted onto a dictionary.

    Because the generator is linear in the field, L_(f + g u) psi =
    L_f psi + u L_g psi, and fitting each term with gedmd gives

        z' = z A + sum_j u_j z B_j,      z = psi(x)  (row vector)

    a bilinear system in the lifted coordinates. It is exact when span(psi)
    is invariant under f and every g_j, and is the standard route to Koopman
    model-predictive and bilinear control.
    """

    def __init__(self, f, gs, psi, X, args=None):
        self.psi = psi
        Psi = _evaluate(psi, X, args)
        self.drift = _fit(Psi, _evaluate(L_d(psi, f), X, args), psi, "generator", X)
        self.inputs = [_fit(Psi, _evaluate(L_d(psi, g), X, args), psi, "generator", X) for g in gs]
        self.A = self.drift.matrix
        self.B = [k.matrix for k in self.inputs]
        self.residual = max([self.drift.residual] + [k.residual for k in self.inputs])
        C, *_ = onp.linalg.lstsq(Psi, onp.asarray(X, dtype=float), rcond=None)
        self.readout = C   # x ~ z C

    def lift(self, x, args=None):
        return onp.asarray(self.psi(x, args), dtype=float)

    def field(self, z, u):
        """dz/dt for lifted state z under input vector u."""
        M = self.A + sum(uj * Bj for uj, Bj in zip(onp.atleast_1d(u), self.B))
        return z @ M

    def simulate(self, x0, u, t):
        """Lift x0, integrate the lifted system under u(t), and read the state back out.

        u is a callable t -> input vector (or a constant). For a constant input
        the lifted system is linear and is solved exactly with a matrix
        exponential; otherwise it is integrated with scipy.
        """
        from scipy.integrate import solve_ivp
        from scipy.linalg import expm

        z0 = self.lift(x0)
        ts = onp.asarray(t, dtype=float)
        if not callable(u):
            M = self.A + sum(uj * Bj for uj, Bj in zip(onp.atleast_1d(u), self.B))
            Z = onp.array([z0 @ expm(M * ti) for ti in ts])
        else:
            sol = solve_ivp(lambda ti, z: self.field(z, u(ti)), (ts[0], ts[-1]), z0,
                            t_eval=ts, rtol=1e-10, atol=1e-12)
            Z = sol.y.T
        return Z @ self.readout
