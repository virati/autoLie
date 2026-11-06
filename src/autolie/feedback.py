"""
Feedback linearization for control-affine systems  x' = f(x) + g(x) u,  y = h(x).

Everything here is built from lie_lib's two operators, so it works on any
autograd-differentiable f, g, h written as operables taking (x, args).

Sign convention. This module uses the control-texts' bracket (Isidori, Khalil):

    ad_f g = [f, g] = (dg/dx) f - (df/dx) g  =  lie_lib.bracket_field(g, f)

lie_lib.bracket_field keeps autoLie's original order, which is the negative.
Only single-input, single-output for now.

@author: virati
"""

import autograd.numpy as np

from autolie.lie_lib import L_d, bracket_field
from autolie.op_plus import operable


def ad(f, g, k=1):
    """ad_f^k g: g bracketed with f, k times. ad_f^0 g = g."""
    for _ in range(k):
        g = bracket_field(g, f)
    return g


def relative_degree(f, g, h, x, args=None, max_order=None, tol=1e-9):
    """The relative degree of y = h(x) at the point x.

    The smallest r with L_g L_f^(r-1) h(x) != 0: how many times y has to be
    differentiated before u shows up. Returns None if that never happens within
    max_order (default: the state dimension), which means the output cannot be
    linearized from this point.
    """
    n = len(x) if max_order is None else max_order
    Lfh = h
    for r in range(1, n + 1):
        if abs(L_d(Lfh, g)(x, args)) > tol:
            return r
        Lfh = L_d(Lfh, f)
    return None


def linearizing_control(f, g, h, r):
    """The input-output linearizing feedback for relative degree r.

    Returns u(x, v, args) = (v - L_f^r h(x)) / (L_g L_f^(r-1) h(x)), under which
    the r-th derivative of y is exactly v. Undefined where the denominator
    vanishes, which is where the relative degree is not r.
    """
    Lfr = L_d(h, f, order=r)
    LgLf = L_d(L_d(h, f, order=r - 1), g) if r > 1 else L_d(h, g)

    def u(x, v, args=None):
        return (v - Lfr(x, args)) / LgLf(x, args)

    return u


def closed_loop(f, g, u):
    """The field x' = f(x) + g(x) u(x, v) for a fixed external input v (taken from args)."""
    return operable(lambda x, v: f(x, None) + g(x, None) * u(x, v))


def controllability_matrix(f, g, x, args=None, n=None):
    """Columns g, ad_f g, ..., ad_f^(n-1) g evaluated at x.

    Full rank n is the accessibility condition for exact (full-state)
    feedback linearization; involutivity of the first n-1 columns is the other.
    """
    n = len(x) if n is None else n
    return np.stack([ad(f, g, k)(x, args) for k in range(n)], axis=1)


def is_involutive(fields, x, args=None, tol=1e-8):
    """Whether span{fields} is closed under the bracket at x.

    Every [X_i, X_j] must lie in the span of the X_k, checked as a rank test.
    This is a pointwise check; involutivity on a neighbourhood needs it to hold
    at every point there.
    """
    D = np.stack([X(x, args) for X in fields], axis=1)
    rank = np.linalg.matrix_rank(D, tol=tol)
    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            b = bracket_field(fields[i], fields[j])(x, args)
            if np.linalg.matrix_rank(np.column_stack([D, b]), tol=tol) > rank:
                return False
    return True


def is_exactly_linearizable(f, g, x, args=None, tol=1e-8):
    """Full-state feedback linearizability at x (single input).

    True when {g, ad_f g, ..., ad_f^(n-1) g} has rank n and
    {g, ..., ad_f^(n-2) g} is involutive. Then an output with relative degree
    n exists, and linearizing_control on it linearizes the whole state.
    """
    n = len(x)
    C = controllability_matrix(f, g, x, args, n)
    if np.linalg.matrix_rank(C, tol=tol) < n:
        return False
    return is_involutive([ad(f, g, k) for k in range(n - 1)], x, args, tol)
