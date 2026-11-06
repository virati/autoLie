"""Feedback linearization on systems whose answers are known in closed form."""

import autograd.numpy as anp
import numpy as onp
import pytest

from autolie.feedback import (
    ad,
    closed_loop,
    controllability_matrix,
    is_exactly_linearizable,
    is_involutive,
    linearizing_control,
    relative_degree,
)
from autolie.lie_lib import L_d
from autolie.op_plus import operable

rng = onp.random.default_rng(1)

# Pendulum: x1' = x2, x2' = -sin(x1) - 0.3 x2 + u, y = x1. Relative degree 2.
pend_f = operable(lambda x, a: anp.array([x[1], -anp.sin(x[0]) - 0.3 * x[1]]))
pend_g = operable(lambda x, a: anp.array([0.0, 1.0]))
pend_h = operable(lambda x, a: x[0])

# A chain with a nonlinear tail: x1' = x2 + x3^2, x2' = x3, x3' = -x1^3 + u.
chain_f = operable(lambda x, a: anp.array([x[1] + x[2] ** 2, x[2], -x[0] ** 3]))
chain_g = operable(lambda x, a: anp.array([0.0, 0.0, 1.0]))


def test_ad_against_hand_computation():
    # ad_f g for the pendulum: (dg/dx) f - (df/dx) g = -(df/dx) e2 = (-1, 0.3).
    p = anp.array([0.4, -1.1])
    assert ad(pend_f, pend_g)(p, None) == pytest.approx([-1.0, 0.3])
    assert ad(pend_f, pend_g, 0)(p, None) == pytest.approx([0.0, 1.0])


@pytest.mark.parametrize(
    "h, r",
    [
        (operable(lambda x, a: x[2]), 1),
        (operable(lambda x, a: x[1]), 2),
        # L_f x1 = x2 + x3^2, so L_g L_f x1 = 2 x3: degree 2 away from x3 = 0.
        (operable(lambda x, a: x[0]), 2),
    ],
)
def test_relative_degree(h, r):
    p = rng.normal(size=3)
    assert relative_degree(chain_f, chain_g, h, p) == r


def test_relative_degree_none_when_input_never_reaches_output():
    f = operable(lambda x, a: anp.array([-x[0], x[1]]))
    g = operable(lambda x, a: anp.array([0.0, 1.0]))
    h = operable(lambda x, a: x[0])
    assert relative_degree(f, g, h, anp.array([0.3, 0.2])) is None


def test_linearizing_control_makes_the_output_a_double_integrator():
    u = linearizing_control(pend_f, pend_g, pend_h, 2)
    for _ in range(5):
        p, v = rng.normal(size=2), rng.normal()
        F = closed_loop(pend_f, pend_g, u)
        # y'' = L_F^2 h under the closed loop must equal v exactly.
        assert L_d(pend_h, F, order=2)(p, v) == pytest.approx(v, rel=1e-9, abs=1e-9)


def test_pendulum_is_exactly_linearizable():
    p = rng.normal(size=2)
    assert onp.linalg.matrix_rank(controllability_matrix(pend_f, pend_g, p)) == 2
    assert is_exactly_linearizable(pend_f, pend_g, p)


def test_involutivity():
    # Rotations about z and translation along z commute up to the span: involutive.
    rot = operable(lambda x, a: anp.array([-x[1], x[0], 0.0]))
    tz = operable(lambda x, a: anp.array([0.0, 0.0, 1.0]))
    assert is_involutive([rot, tz], anp.array([1.0, 0.5, 0.2]))
    # The Heisenberg pair: [X, Y] = -2 e_z (here), outside span{X, Y}.
    X = operable(lambda x, a: anp.array([1.0, 0.0, -x[1]]))
    Y = operable(lambda x, a: anp.array([0.0, 1.0, x[0]]))
    assert not is_involutive([X, Y], anp.array([0.3, -0.2, 0.1]))


# --- normal form, zero dynamics, MIMO -------------------------------------------

from autolie.feedback import (  # noqa: E402
    decoupling_matrix,
    mimo_closed_loop,
    mimo_linearizing_control,
    normal_coordinates,
    vector_relative_degree,
    zero_dynamics,
)


def test_normal_coordinates_become_an_integrator_chain():
    z = normal_coordinates(pend_f, pend_h, 2)
    u = linearizing_control(pend_f, pend_g, pend_h, 2)
    for _ in range(4):
        p, v = rng.normal(size=2), rng.normal()
        F = closed_loop(pend_f, pend_g, u)
        zdot = L_d(z, F)(p, v)
        assert zdot == pytest.approx([z(p, None)[1], v], rel=1e-9, abs=1e-9)


def test_zero_dynamics_keep_the_output_at_zero():
    # Chain system, y = x1 (degree 2). On {x1 = 0, x2 + x3^2 = 0} the zero
    # dynamics field must not move either coordinate off zero.
    h = operable(lambda x, a: x[0])
    F0 = zero_dynamics(chain_f, chain_g, h, 2)
    z = normal_coordinates(chain_f, h, 2)
    for x3 in [0.4, -0.7, 1.3]:
        p = anp.array([0.0, -(x3**2), x3])
        assert z(p, None) == pytest.approx([0.0, 0.0], abs=1e-12)
        assert L_d(z, F0)(p, None) == pytest.approx([0.0, 0.0], abs=1e-9)


# Two inputs, two outputs: x1' = x2, x2' = -x1 + x2 x3 + u1 + x1 u2, x3' = -x3 + u2.
mf = operable(lambda x, a: anp.array([x[1], -x[0] + x[1] * x[2], -x[2]]))
mg = [
    operable(lambda x, a: anp.array([0.0, 1.0, 0.0])),
    operable(lambda x, a: anp.array([0.0, x[0], 1.0])),
]
mh = [operable(lambda x, a: x[0]), operable(lambda x, a: x[2])]


def test_vector_relative_degree_and_decoupling_matrix():
    p = anp.array([0.7, -0.2, 0.5])
    assert vector_relative_degree(mf, mg, mh, p) == [2, 1]
    # A = [[L_g1 x2, L_g2 x2], [L_g1 x3, L_g2 x3]] = [[1, x1], [0, 1]]
    assert decoupling_matrix(mf, mg, mh, [2, 1], p).ravel() == pytest.approx([1.0, 0.7, 0.0, 1.0])


def test_mimo_control_decouples_the_outputs():
    u = mimo_linearizing_control(mf, mg, mh, [2, 1])
    F = mimo_closed_loop(mf, mg, u)
    for _ in range(4):
        p, v = rng.normal(size=3), rng.normal(size=2)
        # y1'' = v1 and y2' = v2, each from its own input alone.
        assert L_d(mh[0], F, order=2)(p, v) == pytest.approx(v[0], rel=1e-9, abs=1e-9)
        assert L_d(mh[1], F)(p, v) == pytest.approx(v[1], rel=1e-9, abs=1e-9)
