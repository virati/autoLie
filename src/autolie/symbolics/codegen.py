"""
Closed-form Lie brackets, and code generation for them.

lie_lib computes brackets numerically with autograd, which is the right tool for
analysis but cannot hand a bracket to anything that is not Python. This module
does the same algebra in sympy so the result can be printed as source code,
for example as a GLSL expression to integrate on the GPU.

    from autolie.symbolics.codegen import symbolize, bracket, glsl

    x, y, z = sym.symbols("x y z")
    f = symbolize(lorenz_rhs)(x, y, z, sigma=10, rho=28, beta=8/3)
    g = symbolize(rossler_rhs)(x, y, z, a=0.2, b=0.2, c=5.7)
    print([glsl(c) for c in bracket(f, g, (x, y, z))])

The convention matches lie_lib.L_bracket: [f, g] = (df/dx) g - (dg/dx) f.
Check a generated bracket against lie_lib.bracket_field before trusting it.

@author: virati
"""

import types

import sympy as sym
from sympy.printing.glsl import GLSLPrinter


class _SympyNumpy:
    """Stands in for `numpy` inside a function being traced symbolically.

    Only elementwise maths is mapped. Anything else (control flow on values,
    reductions, random numbers) raises, which is the honest answer: such a
    function has no single closed form to differentiate.
    """

    pi = sym.pi
    e = sym.E
    sin, cos, tan = sym.sin, sym.cos, sym.tan
    arcsin, arccos, arctan = sym.asin, sym.acos, sym.atan
    sinh, cosh, tanh = sym.sinh, sym.cosh, sym.tanh
    exp, log, sqrt, cbrt = sym.exp, sym.log, sym.sqrt, sym.cbrt
    abs = absolute = sym.Abs
    sign = sym.sign
    floor = sym.floor
    arctan2 = staticmethod(lambda y, x: sym.atan2(y, x))
    power = staticmethod(lambda a, b: a**b)
    square = staticmethod(lambda a: a**2)
    mod = staticmethod(lambda a, b: sym.Mod(a, b))
    minimum = staticmethod(lambda a, b: sym.Min(a, b))
    maximum = staticmethod(lambda a, b: sym.Max(a, b))
    heaviside = staticmethod(lambda a, h0=0: sym.Heaviside(a, h0))
    array = asarray = staticmethod(lambda a, *_, **__: list(a))

    def __getattr__(self, name):
        raise NotImplementedError(f"numpy.{name} has no symbolic stand-in")


def symbolize(fn, np_name="np"):
    """Return a copy of `fn` whose global `np_name` is a sympy-backed numpy.

    Call the copy with sympy symbols to get its output as expressions. Works on
    plain numpy-style functions; for a numba dispatcher, pass `fn.py_func`.
    """
    fn = getattr(fn, "py_func", fn)
    scope = dict(fn.__globals__)
    scope[np_name] = _SympyNumpy()
    return types.FunctionType(fn.__code__, scope, fn.__name__, fn.__defaults__, fn.__closure__)


def bracket(f, g, coords):
    """[f, g] = (df/dx) g - (dg/dx) f for fields given as sequences of sympy expressions."""
    X = sym.Matrix(coords)
    F, G = sym.Matrix(list(f)), sym.Matrix(list(g))
    B = F.jacobian(X) * G - G.jacobian(X) * F
    return [sym.expand(c) for c in B]


class _GLSLES(GLSLPrinter):
    """GLSL ES 3.00: every literal a float, integer powers as products.

    pow(x, 2.0) is undefined for negative x in GLSL, and an attractor spends
    half its life at negative x, so small integer powers are multiplied out.
    """

    def __init__(self, digits=7):
        super().__init__({"contract": False})
        self.digits = digits

    def _float(self, v):
        s = f"{float(v):.{self.digits}g}"
        return s if any(c in s for c in ".en") else s + ".0"

    def _print_Integer(self, expr):
        return self._float(expr)

    _print_Rational = _print_Float = _print_Integer

    def _print_NegativeOne(self, expr):
        return "-1.0"

    def _print_Pow(self, expr):
        base, ex = expr.as_base_exp()
        if ex.is_Integer and 0 < abs(int(ex)) <= 6:
            b = self.parenthesize(base, sym.printing.precedence.PRECEDENCE["Mul"] + 1)
            prod = " * ".join([b] * abs(int(ex)))
            return prod if ex > 0 else f"1.0 / ({prod})"
        if ex == sym.S.Half:
            return f"sqrt({self._print(base)})"
        if ex == -sym.S.Half:
            return f"inversesqrt({self._print(base)})"
        return f"pow({self._print(base)}, {self._print(ex)})"

    def _print_Abs(self, expr):
        return f"abs({self._print(expr.args[0])})"

    def _print_sign(self, expr):
        return f"sign({self._print(expr.args[0])})"

    def _print_Heaviside(self, expr):
        return f"step(0.0, {self._print(expr.args[0])})"

    def _print_atan2(self, expr):
        return f"atan({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_Min(self, expr):
        a, *rest = expr.args
        return f"min({self._print(a)}, {self._print(sym.Min(*rest)) if len(rest) > 1 else self._print(rest[0])})"

    def _print_Max(self, expr):
        a, *rest = expr.args
        return f"max({self._print(a)}, {self._print(sym.Max(*rest)) if len(rest) > 1 else self._print(rest[0])})"

    def _print_Mod(self, expr):
        return f"mod({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_DiracDelta(self, expr):
        raise NotImplementedError("bracket contains a DiracDelta (derivative of a step)")

    _print_Derivative = _print_Subs = _print_DiracDelta


def glsl(expr, digits=7):
    """Print one scalar sympy expression as GLSL ES 3.00 source."""
    return _GLSLES(digits).doprint(expr)
