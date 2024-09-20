import operator
import re
import sys
import tomllib
import typing
from collections.abc import Iterable, Sequence
from itertools import pairwise, starmap
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from mip import LinExpr, Model, Var, maximize, xsum
from numpy import isneginf, isposinf

if typing.TYPE_CHECKING:
    from contextlib import suppress

    with suppress(ImportError):
        import pulp as pl

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


def add_line(m: Model, p1: Point, p2: Point, x: Var, y: Var, *, under: bool) -> None:
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


def add_lines_conv(m: Model, curve: np.ndarray, x: Var, y: Var, *, upward: bool = False):
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
        add_line(m, p1, p2, x, y, under=upward)


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


def model2str(m: Model):
    with TemporaryDirectory() as dir_:
        file_name = Path(dir_) / "dummy.lp"
        m.write(str(file_name))
        return file_name.read_text("utf-8")


def show_model(m: Model, out=sys.stdout):
    """Show LP format

    :param m: Model
    :param out: Output stream, defaults to sys.stdout
    """
    if not m.vars:
        m.add_var()  # `m.write` will error if there is no variable
    print(model2str(m), file=out)


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
        lb = v.lb if v.lb > -1e308 else -np.inf  # noqa: PLR2004
        ub = v.ub if v.ub < 1e308 else np.inf  # noqa: PLR2004
        if not np.isinf([lb, ub]).all():
            yield [*list(e), lb, ub]


def scipy_milp(m: Model, options: dict[str, Any] | None = None):
    """Solve by scipy.milp from mip.Model

    :param m: mip.Model
    :param options: options of scipy.milp, defaults to None
    :return: result of scipy.milp
    """
    from scipy.optimize import Bounds, LinearConstraint, milp  # noqa: PLC0415

    args: dict[str, Any] = {"options": options}
    args["c"] = np.array([v.obj for v in m.vars])
    if m.sense != "MIN":
        args["c"] = -args["c"]
    args["integrality"] = [int(v.var_type != "C") for v in m.vars]
    args["bounds"] = Bounds()
    alu = np.array(list(_cnst(m)))
    args["constraints"] = LinearConstraint(alu[:, :-2], alu[:, -2], alu[:, -1])
    return milp(**args)


def model2toml(m: Model) -> str:
    cons = {re.sub(r"[ -/:-@\[-`{-~]", "_", co.name): co for co in m.constrs}
    assert len({va.name for va in m.vars}) == len(m.vars), "Variable names must be unique"
    assert len(cons) == len(m.constrs), "Constraint names must be unique"
    lst = [f"name = {m.name!r}", f"sense = {m.sense!r}", "[vars]"]
    for va in m.vars:
        lb = "-inf" if va.lb <= -sys.float_info.max else f"{va.lb:g}"
        ub = "inf" if va.ub >= sys.float_info.max else f"{va.ub:g}"
        lst.append(f"{va.name} = [{lb}, {ub}, {va.obj:g}, {va.var_type!r}]")
    lst.append("[constrs]")
    for name, co in cons.items():
        variables = [v.name for v in co.expr.expr]
        coeffs = "[" + ", ".join(f"{i:g}" for i in co.expr.expr.values()) + "]"
        lst.append(f"{name} = [{variables}, {coeffs}, {co.expr.const:g}, {co.expr.sense!r}]")
    return "\n".join(lst)


def write_toml(m: Model, filename: str, encoding: str | None = None) -> None:
    Path(filename).write_text(model2toml(m), encoding)


def toml2model(data: dict[str, Any]) -> Model:
    m = Model(name=data.get("name", ""), sense=data["sense"])
    for name, args in data["vars"].items():
        m.add_var(name, *args)
    for name, (variables, coeffs, const, sense) in data["constrs"].items():
        _variables = [m.vars[s] for s in variables]
        m.add_constr(LinExpr(_variables, coeffs, const, sense), name)
    return m


def read_toml(filename: str, encoding: str | None = None) -> Model:
    data = tomllib.loads(Path(filename).read_text(encoding))
    return toml2model(data)


def pulp_model2toml(m: "pl.LpProblem") -> str:
    import pulp as pl  # noqa: PLC0415

    vars_ = m.variables()
    cons = {re.sub(r"[ -/:-@\[-`{-~]", "_", name): co for name, co in m.constraints.items()}
    assert len({va.name for va in vars_}) == len(vars_), "Variable names must be unique"
    assert len(cons) == len(m.constraints), "Constraint names must be unique"
    lst = [f"name = {m.name!r}", f"sense = {'MIN' if m.sense == 1 else 'MAX'!r}", "[vars]"]
    for va in vars_:
        lb = "-inf" if va.lowBound is None else f"{va.lowBound:g}"
        ub = "inf" if va.upBound is None else f"{va.upBound:g}"
        obj = m.objective.get(va, 0)
        lst.append(f"{va.name} = [{lb}, {ub}, {obj:g}, {va.cat[0]!r}]")
    lst.append("[constrs]")
    for name, co in cons.items():
        variables = [va.name for va in co]
        coeffs = "[" + ", ".join(f"{i:g}" for i in co.values()) + "]"
        sense = pl.LpConstraintSenses[co.sense][0]
        lst.append(f"{name} = [{list(variables)}, {coeffs}, {co.constant:g}, {sense!r}]")
    return "\n".join(lst)


def toml2pulp_model(data: dict[str, Any]) -> "pl.LpProblem":
    import pulp as pl  # noqa: PLC0415

    cats = {"B": pl.LpBinary, "C": pl.LpContinuous, "I": pl.LpInteger}
    senses: dict[str, int] = {"=": 0, "<": -1, ">": 1}
    sense = 1 if data["sense"] == "MIN" else -1
    m = pl.LpProblem(name=data.get("name", ""), sense=sense)
    e = pl.LpAffineExpression()
    vars_ = {}
    for name, (lb, ub, obj, cat) in data["vars"].items():
        _lb = None if isneginf(lb) else lb
        _ub = None if isposinf(ub) else ub
        vars_[name] = v = pl.LpVariable(name, _lb, _ub, cat=cats[cat])
        if obj:
            e.addterm(v, obj)
    m.objective = e
    for name, (variables, coeffs, const, sense_) in data["constrs"].items():
        _variables = [vars_[s] for s in variables]
        e = pl.lpDot(coeffs, _variables)
        sense = senses[sense_]
        m.addConstraint(pl.LpConstraint(e, sense=sense, rhs=-const), name)
    return m
