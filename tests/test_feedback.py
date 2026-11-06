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
