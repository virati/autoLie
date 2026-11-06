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
