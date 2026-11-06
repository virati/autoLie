"""Koopman approximations on systems with known finite-dimensional spectra."""

import autograd.numpy as anp
import numpy as onp
import pytest
from scipy.linalg import expm

from autolie.koopman import edmd, gedmd, polynomial_dictionary
from autolie.lie_lib import L_d
from autolie.op_plus import operable

rng = onp.random.default_rng(2)
MU, LAM = -0.3, -1.7

# x0' = mu x0, x1' = lam (x1 - x0^2): span{x0, x1, x0^2} is Koopman-invariant,
# with generator eigenvalues mu, lam and 2 mu.
f = operable(lambda x, a: anp.array([MU * x[0], LAM * (x[1] - x[0] ** 2)]))
psi = operable(lambda x, a: anp.array([x[0], x[1], x[0] ** 2]))


def test_gedmd_recovers_the_exact_spectrum():
    K = gedmd(f, psi, rng.normal(size=(40, 2)))
    assert K.residual < 1e-12
    assert sorted(K.eigenvalues.real) == pytest.approx(sorted([MU, LAM, 2 * MU]))


def test_eigenfunctions_satisfy_the_eigen_equation():
    K = gedmd(f, psi, rng.normal(size=(40, 2)))
    for i, lam in enumerate(K.eigenvalues):
        phi = K.eigenfunction(i)
        phi_op = operable(lambda x, a, phi=phi: anp.dot(psi(x, a), K.eigenvectors[:, i].real))
        for p in rng.normal(size=(5, 2)):
            assert L_d(phi_op, f)(p, None) == pytest.approx(lam.real * phi(p).real, abs=1e-10)


def test_edmd_on_a_linear_flow_map():
    A = onp.array([[-0.2, 1.0], [-1.0, -0.2]])
    dt = 0.1
    X = rng.normal(size=(30, 2))
    Y = X @ expm(A * dt).T
    lin, _ = polynomial_dictionary(2, 1, constant=False)
    K = edmd(lin, X, Y)
    assert sorted(K.eigenvalues, key=lambda z: z.imag) == pytest.approx(
        sorted(onp.exp(onp.linalg.eigvals(A) * dt), key=lambda z: z.imag)
    )


def test_polynomial_dictionary():
    psi2, labels = polynomial_dictionary(2, 2)
    assert labels == ["1", "x0", "x1", "x0^2", "x0*x1", "x1^2"]
    assert psi2(anp.array([2.0, 3.0]), None) == pytest.approx([1, 2, 3, 4, 6, 9])
    # The generator of a dictionary is differentiable through autograd.
    Lpsi = L_d(psi2, f)(anp.array([2.0, 3.0]), None)
    assert Lpsi[0] == 0.0 and Lpsi[1] == pytest.approx(MU * 2)


# --- modes, prediction, control ---------------------------------------------------

from autolie.koopman import BilinearLift  # noqa: E402


def exact(x0, t):
    # The slow-manifold system in closed form: x1 relaxes onto b x0^2.
    b = LAM / (LAM - 2 * MU)
    a, c = x0
    return onp.array([a * onp.exp(MU * t), (c - b * a**2) * onp.exp(LAM * t) + b * a**2 * onp.exp(2 * MU * t)])


def test_mode_decomposition_predicts_the_trajectory():
    K = gedmd(f, psi, rng.normal(size=(40, 2)))
    assert K.state_residual < 1e-12
    x0 = onp.array([0.8, -0.4])
    for t in [0.0, 0.5, 2.0, 5.0]:
        assert K.predict(x0, t) == pytest.approx(exact(x0, t), abs=1e-9)


def test_flow_map_prediction_steps():
    dt = 0.1
    X = rng.normal(size=(40, 2))
    Y = onp.array([exact(x, dt) for x in X])
    K = edmd(psi, X, Y)
    x0 = onp.array([0.5, 0.3])
    assert K.predict(x0, [0, 1, 7]) == pytest.approx(
        onp.array([exact(x0, k * dt) for k in [0, 1, 7]]), abs=1e-9
    )


def test_bilinear_lift_reproduces_the_controlled_system():
    from scipy.integrate import solve_ivp

    # Drive x0 directly: g = (1, 0). With a constant in the dictionary,
    # span{1, x0, x1, x0^2} is invariant under both f and g.
    g = operable(lambda x, a: anp.array([1.0, 0.0]))
    psi_c = operable(lambda x, a: anp.array([1.0 + 0.0 * x[0], x[0], x[1], x[0] ** 2]))
    lift = BilinearLift(f, [g], psi_c, rng.normal(size=(40, 2)))
    assert lift.residual < 1e-12

    x0, ts = onp.array([0.6, -0.2]), onp.linspace(0, 3, 7)
    for u in [0.4, lambda t: onp.array([onp.sin(2 * t)])]:
        uf = u if callable(u) else (lambda t, u=u: onp.array([u]))
        true = solve_ivp(lambda t, x: onp.asarray(f(x, None)) + onp.asarray(g(x, None)) * uf(t)[0],
                         (0, 3), x0, t_eval=ts, rtol=1e-11, atol=1e-12).y.T
        assert lift.simulate(x0, u, ts) == pytest.approx(true, abs=1e-6)
