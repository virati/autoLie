"""Checks autoLie's numerical operators against sympy, and against the algebra
a Lie bracket has to satisfy whatever it is computed with."""

import autograd.numpy as anp
import numpy as onp
import pytest
import sympy as sym

from autolie.lie_lib import L_bracket, L_d, L_dot, bracket_field
from autolie.op_plus import operable

X = sym.symbols("x0 x1 x2")
rng = onp.random.default_rng(0)
POINTS = [rng.normal(size=3) for _ in range(5)]


def field(exprs):
    """The same field twice: as sympy expressions and as an autograd operable."""
    fn = sym.lambdify([X], list(exprs), modules=[{"array": anp.array}, anp])
    return list(exprs), operable(lambda x, args: anp.array(fn(x)))


def scalar(expr):
    fn = sym.lambdify([X], expr, modules=anp)
    return expr, operable(lambda x, args: fn(x))


def sym_lie(h, f):
    return sum(sym.diff(h, xi) * fi for xi, fi in zip(X, f))


def sym_bracket(f, g, coords):
    Xm = sym.Matrix(coords)
    F_, G_ = sym.Matrix(f), sym.Matrix(g)
    return list(F_.jacobian(Xm) * G_ - G_.jacobian(Xm) * F_)


def at(expr, p):
    return float(expr.subs(dict(zip(X, p))))


x, y, z = X
F_SYM, F = field([y * z - x, sym.sin(x) - z**2, x * y + sym.exp(-z * z)])
G_SYM, G = field([x**3, y - x * z, sym.cos(y) * z])
H_SYM, H = field([z, -y * y, x + y])
h_sym, h = scalar(x**2 * y + z)


def test_directional_derivative_by_hand():
    # grad h = (2xy, x^2, 1) = (4, 1, 1) at (1, 2, 3); f = (y, -x, xz) = (2, -1, 3).
    _, f = field([y, -x, x * z])
    p = anp.array([1.0, 2.0, 3.0])
    assert L_d(h, f)(p, None) == pytest.approx(10.0)
    assert L_dot(h, f)(p, None) == pytest.approx(10.0)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_directional_derivative_orders(order):
    expected = h_sym
    for _ in range(order):
        expected = sym_lie(expected, F_SYM)
    Lh = L_d(h, F, order=order)
    for p in POINTS:
        assert Lh(p, None) == pytest.approx(at(expected, p), rel=1e-9, abs=1e-9)


def test_directional_derivative_of_a_vector_field():
    # Componentwise: (dG/dx) F.
    LG = L_d(G, F)
    for p in POINTS:
        expected = [at(sym_lie(gi, F_SYM), p) for gi in G_SYM]
        assert LG(p, None) == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_bracket_by_hand():
    # The example in lie_lib's own __main__: f1 = (-y^2 + z, -x^3, -z^2 + y),
    # g1 = (x, 0, y) at (1, -2, 3). Jf g = (-2, -3, 12), Jg f = (-1, 0, -1).
    _, f1 = field([-(y**2) + z, -(x**3), -(z**2) + y])
    _, g1 = field([x, sym.Integer(0), y])
    p = anp.array([1.0, -2.0, 3.0])
    assert bracket_field(f1, g1)(p, None) == pytest.approx([-1.0, -3.0, 13.0])
    assert L_bracket(f1, g1, args=[p, None]) == pytest.approx([-1.0, -3.0, 13.0])


def test_bracket_matches_sympy():
    B_sym = sym_bracket(F_SYM, G_SYM, X)
    B = bracket_field(F, G)
    for p in POINTS:
        assert B(p, None) == pytest.approx([at(b, p) for b in B_sym], rel=1e-9, abs=1e-9)


def test_bracket_of_linear_fields_is_the_commutator():
    A = rng.normal(size=(3, 3))
    C = rng.normal(size=(3, 3))
    f = operable(lambda v, args: anp.dot(A, v))
    g = operable(lambda v, args: anp.dot(C, v))
    for p in POINTS:
        # (df/dx) g - (dg/dx) f = A C x - C A x
        assert bracket_field(f, g)(p, None) == pytest.approx((A @ C - C @ A) @ p)


def test_bracket_is_antisymmetric():
    for p in POINTS:
        assert bracket_field(F, G)(p, None) == pytest.approx(-bracket_field(G, F)(p, None))


def test_jacobi_identity():
    total = lambda p: (  # noqa: E731
        bracket_field(F, bracket_field(G, H))(p, None)
        + bracket_field(G, bracket_field(H, F))(p, None)
        + bracket_field(H, bracket_field(F, G))(p, None)
    )
    for p in POINTS:
        assert total(p) == pytest.approx(onp.zeros(3), abs=1e-8)


def test_bracket_acts_as_commutator_of_lie_derivatives():
    # L_[f,g] h = L_g L_f h - L_f L_g h in this sign convention, since
    # [f, g] = (df/dx) g - (dg/dx) f is minus the derivation commutator.
    lhs = L_d(h, bracket_field(F, G))
    for p in POINTS:
        rhs = L_d(L_d(h, F), G)(p, None) - L_d(L_d(h, G), F)(p, None)
        assert lhs(p, None) == pytest.approx(rhs, rel=1e-9, abs=1e-9)
