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
    """A fitted matrix on a dictionary, with its spectrum and eigenfunctions."""

    def __init__(self, psi, M, residual):
        self.psi = psi
        self.matrix = M
        self.residual = residual
        self.eigenvalues, self.eigenvectors = onp.linalg.eig(M)

    def eigenfunction(self, i):
        xi = self.eigenvectors[:, i]
        return lambda x, args=None: onp.dot(onp.asarray(self.psi(x, args)), xi)


def _evaluate(fn, X, args=None):
    return onp.array([onp.asarray(fn(x, args), dtype=float) for x in X])


def _fit(Psi, target, psi):
    M, *_ = onp.linalg.lstsq(Psi, target, rcond=None)
    resid = onp.linalg.norm(Psi @ M - target) / max(onp.linalg.norm(target), 1e-300)
    return KoopmanApprox(psi, M, resid)


def gedmd(f, psi, X, args=None):
    """Generator EDMD: fit L_f psi ~ psi M on the sample points X (N x n).

    The derivative of the dictionary along f is computed exactly with L_d,
    so the only approximation is the projection onto span(psi). A relative
    residual near zero means the dictionary is (nearly) Koopman-invariant.
    """
    return _fit(_evaluate(psi, X, args), _evaluate(L_d(psi, f), X, args), psi)


def edmd(psi, X, Y, args=None):
    """Snapshot EDMD: fit psi(Y) ~ psi(X) K from pairs Y[i] = F(X[i])."""
    return _fit(_evaluate(psi, X, args), _evaluate(psi, Y, args), psi)
