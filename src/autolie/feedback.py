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


# --- normal form and zero dynamics ----------------------------------------------


def normal_coordinates(f, h, r):
    """z(x) = (h, L_f h, ..., L_f^(r-1) h), the linearizing coordinates for y = h.

    Under linearizing_control these evolve as a chain of r integrators,
    z_i' = z_(i+1) and z_r' = v.
    """
    terms = [h] + [L_d(h, f, order=k) for k in range(1, r)]
    return operable(lambda x, args: np.array([np.reshape(c(x, args), ()) for c in terms]))


def zero_dynamics(f, g, h, r):
    """The internal dynamics left over when y is held at zero.

    Returns the field x' = f(x) + g(x) u*(x), with u* the linearizing control at
    v = 0. It leaves the manifold {z(x) = 0} invariant, and its restriction to
    that (n - r)-dimensional manifold is the zero dynamics: the part of the
    state the output cannot see. When it is unstable the output can be tracked
    only while the rest of the state runs away.
    """
    u = linearizing_control(f, g, h, r)
    return operable(lambda x, args: f(x, None) + g(x, None) * u(x, 0.0))


# --- several inputs and outputs -------------------------------------------------


def _lgf(f, gj, h, k):
    """L_gj L_f^k h."""
    return L_d(L_d(h, f, order=k) if k else h, gj)


def vector_relative_degree(f, gs, hs, x, args=None, max_order=None, tol=1e-9):
    """(r_1, ..., r_p): for each output, the first r with some L_gj L_f^(r-1) h_i != 0.

    gs and hs are lists of operables, one per input and per output. An entry is
    None when that output never sees any input within max_order.
    """
    n = len(x) if max_order is None else max_order
    out = []
    for h in hs:
        r = None
        for k in range(n):
            if any(abs(_lgf(f, gj, h, k)(x, args)) > tol for gj in gs):
                r = k + 1
                break
        out.append(r)
    return out


def decoupling_matrix(f, gs, hs, rs, x, args=None):
    """A(x)[i, j] = L_gj L_f^(r_i - 1) h_i, the map from inputs to the r_i-th output derivatives."""
    return np.array([[_lgf(f, gj, h, r - 1)(x, args) for gj in gs] for h, r in zip(hs, rs)])


def mimo_linearizing_control(f, gs, hs, rs):
    """u(x, v) = A(x)^-1 (v - b(x)), with b_i = L_f^(r_i) h_i.

    Under it, output i obeys y_i^(r_i) = v_i, each driven by its own v_i alone:
    input-output linearization and decoupling at once. Needs as many outputs as
    inputs and A(x) invertible, which is what a well-defined vector relative
    degree means.
    """
    if len(gs) != len(hs):
        raise ValueError("decoupling needs as many outputs as inputs")
    drift = [L_d(h, f, order=r) for h, r in zip(hs, rs)]

    def u(x, v, args=None):
        A = decoupling_matrix(f, gs, hs, rs, x, args)
        b = np.array([d(x, args) for d in drift])
        return np.linalg.solve(A, np.asarray(v) - b)

    return u


def mimo_closed_loop(f, gs, u):
    """x' = f(x) + sum_j g_j(x) u_j(x, v), with v taken from args."""

    def F(x, v):
        uu = u(x, v)
        out = f(x, None)
        for j, gj in enumerate(gs):
            out = out + gj(x, None) * uu[j]
        return out

    return operable(F)
