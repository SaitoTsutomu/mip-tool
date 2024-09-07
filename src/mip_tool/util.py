import operator
import os
import sys
from collections.abc import Iterable, Sequence
from itertools import pairwise, starmap
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from mip import LinExpr, Model, Var, maximize, xsum

Point = Sequence[float]


def monotone_increasing(it: Iterable):
    """Check monotonous increasing

    :param it: iterable of number
    :return: True if monotonous increasing
    """
    return all(starmap(operator.le, pairwise(it)))


def monotone_decreasing(it: Iterable):
    """Check monotonous decreasing

    :param it: iterable of number
    :return: True if monotonous decreasing
    """
    return all(starmap(operator.ge, pairwise(it)))


def add_line(m: Model, p1: Point, p2: Point, x: Var, y: Var, under: bool) -> None:
    """Add constraint which pass through p1 and p2.

    :param m: Model
    :param p1: Point 1
    :param p2: Point 2
    :param x: Var x
    :param y: Var y
    :param under: 'y <= ...' if True, defaults to True
    """
    if dx := p2[0] - p1[0]:
        const = p1[0] * p2[1] - p2[0] * p1[1]
        cx = p1[1] - p2[1]
        m += LinExpr([x, y], [cx, dx], const, "><"[int(under)])


def add_lines_conv(m: Model, curve: np.ndarray, x: Var, y: Var, upward: bool = False):
    """Add convex piecewise linear constraint

    :param m: Model
    :param curve: Point ndarray
    :param x: Var x
    :param y: Var y
    :param upward: Convex upward if True, defaults to False
    """
    tilt = np.divide(*np.diff(curve, axis=0).T[[1, 0]])
    if upward:
        assert monotone_decreasing(tilt), "Tilt must be decr"
    else:
        assert monotone_increasing(tilt), "Tilt must be incr"
    for p1, p2 in pairwise(curve):
        add_line(m, p1, p2, x, y, upward)


def add_lines(m: Model, curve: np.ndarray, x: Var, y: Var):
    """Add non-convex piecewise linear constraint

    :param m: Model
    :param curve: Point ndarray
    :param x: Var x
    :param y: Var y
    """
    n = curve.shape[0]
    w = m.add_var_tensor((n - 1,), "w")
    z = m.add_var_tensor((n - 2,), "z", var_type="B")
    a, b = curve.T
    m += x == a[0] + xsum(w)
    c = [(b[i + 1] - b[i]) / (a[i + 1] - a[i]) for i in range(n - 1)]
    m += y == b[0] + xsum(c * w)
    for i in range(n - 1):
        if i < n - 2:
            m += (a[i + 1] - a[i]) * z[i] <= w[i]
        m += w[i] <= (a[i + 1] - a[i]) * (1 if i == 0 else z[i - 1])


def show_model(m: Model, out=sys.stdout):
    """Show LP format

    :param m: Model
    :param out: Output stream, defaults to sys.stdout
    """
    if not m.vars:
        m.add_var()  # `m.write` will error if there is no variable
    with TemporaryDirectory() as dir_:
        file_name = os.path.join(dir_, "dummy.lp")
        m.write(file_name)
        with open(file_name) as fp:
            print(fp.read(), file=out)


def random_model(nv: int, nc: int, seed: int | None = None, rtco: float = 0.5, var_type: str = "B"):
    """random model

    :param it: number of variables
    :param nc: number of constraints
    :param seed: seed
    :param rtco: rate of non zero in matrix
    :param var_type: variable type
    """
    rtco = max(1e-3, rtco)
    rnd = np.random.default_rng(seed)
    while True:
        m = Model(solver_name="CBC")
        x = m.add_var_tensor((nv,), "x", var_type=var_type)
        m.objective = maximize(xsum(rnd.integers(20, 40, nv) * x))
        rem = nc
        while rem > 0:
            a = rnd.integers(10, 30, nv)
            a[rnd.random(nv) >= rtco] = 0
            if not a.sum():
                continue
            m += xsum(a * x) <= rnd.integers(a.sum() // 2, a.sum() * 2 // 3)
            rem -= 1
        m.verbose = 0
        m.optimize()
        if not m.status.value:
            return m


def _cnst(m: Model):
    for c in m.constrs:
        e = [c.expr.expr.get(v, 0) for v in m.vars]
        lb = c.rhs if c.expr.sense != "<" else -np.inf
        ub = c.rhs if c.expr.sense != ">" else np.inf
        yield [*e, lb, ub]
    n = len(m.vars)
    for v, e in zip(m.vars, np.eye(n), strict=False):
        lb = v.lb if v.lb > -1e308 else -np.inf
        ub = v.ub if v.ub < 1e308 else np.inf
        if not np.isinf([lb, ub]).all():
            yield [*list(e), lb, ub]


def scipy_milp(m: Model, options: dict[str, Any] | None = None):
    """Solve by scipy.milp from mip.Model

    :param m: mip.Model
    :param options: options of scipy.milp, defaults to None
    :return: result of scipy.milp
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    args: dict[str, Any] = {"options": options}
    args["c"] = np.array([v.obj for v in m.vars])
    if m.sense != "MIN":
        args["c"] = -args["c"]
    args["integrality"] = [int(v.var_type != "C") for v in m.vars]
    args["bounds"] = Bounds()
    alu = np.array(list(_cnst(m)))
    args["constraints"] = LinearConstraint(alu[:, :-2], alu[:, -2], alu[:, -1])
    return milp(**args)
