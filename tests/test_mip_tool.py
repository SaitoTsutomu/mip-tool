import numpy as np
import pandas as pd
import pytest
from mip import INF, Model, OptimizationStatus, maximize, xsum

from mip_tool import add_line, add_lines, add_lines_conv, monotone_decreasing, monotone_increasing
from mip_tool.func import F, addbinvars, addintvars, addvars


def test_monotone_increasing():
    assert monotone_increasing([-1, 3, 3, 4])
    assert not monotone_increasing([3, 3, 2, -1, -1])


def test_monotone_decreasing():
    assert not monotone_decreasing([-1, 3, 3, 4])
    assert monotone_decreasing([3, 3, 2, -1, -1])


def check_add_line(xargs, yargs, p1, p2, under, expect):
    """1制約条件の確認"""
    m = Model()
    x = m.add_var("x", **xargs)
    y = m.add_var("y", **yargs)
    add_line(m, p1, p2, x, y, under)
    m.verbose = 0
    m.optimize()
    assert m.status == OptimizationStatus.OPTIMAL
    assert (x.x, y.x) == expect
    return m


def test_add_line_1():
    """1制約条件の最小化"""
    check_add_line(dict(lb=-INF, ub=-1), dict(obj=1), [1, 2], [3, 1], False, (-1, 3))


def test_add_line_2():
    """1制約条件の最大化"""
    check_add_line(dict(lb=-2), dict(obj=-1), [1, 2], [3, 1], True, (-2, 3.5))


def check_add_lines_conv(xargs, yargs, curve, upward, expect):
    """区分線形近似（凸）の確認"""
    m = Model()
    x = m.add_var("x", **xargs)
    y = m.add_var("y", **yargs)
    add_lines_conv(m, curve, x, y, upward)
    m.verbose = 0
    m.optimize()
    assert m.status == OptimizationStatus.OPTIMAL
    assert (x.x, y.x) == expect
    return m


def test_add_lines_conv_1():
    """区分線形近似（凸）の最小化"""
    curve = np.array([[-3, -5], [-2, -7], [-1, -6]])
    check_add_lines_conv(dict(lb=-INF), dict(obj=1, lb=-INF), curve, False, (-2, -7))


def test_add_lines_conv_2():
    """区分線形近似（凸）の最大化"""
    curve = np.array([[2, 3], [3, 5], [4, 4]])
    check_add_lines_conv(dict(), dict(obj=-1, lb=-INF), curve, True, (3, 5))


def check_add_lines(xargs, yargs, curve, expect):
    """区分線形近似の確認"""
    m = Model()
    x = m.add_var("x", **xargs)
    y = m.add_var("y", **yargs)
    add_lines(m, curve, x, y)
    m.verbose = 0
    m.optimize()
    assert m.status == OptimizationStatus.OPTIMAL
    assert (x.x, y.x) == expect
    return m


def test_add_lines_1():
    """区分線形近似の色々な確認"""
    curve = np.array([[-2, 6], [-1, 7], [2, -2], [4, 5]])
    check_add_lines(dict(lb=-2, ub=-2), dict(), curve, (-2, 6))
    check_add_lines(dict(lb=-INF), dict(obj=-1), curve, (-1, 7))
    check_add_lines(dict(lb=1, ub=1), dict(), curve, (1, 1))
    check_add_lines(dict(), dict(obj=1, lb=-INF), curve, (2, -2))
    check_add_lines(dict(lb=4, ub=4), dict(), curve, (4, 5))


@pytest.fixture
def for_F():
    m = Model()
    x = m.add_var("x", lb=-INF)
    y = m.add_var("y", obj=1, lb=-INF)
    return m, x, y


def check_F(m, x, y, expect):
    m.verbose = 0
    m.optimize()
    assert m.status == OptimizationStatus.OPTIMAL
    assert (x.x, y.x) == expect


def test_F_1(for_F):
    m, x, y = for_F
    y.obj = -1
    m += y <= F([[0, 1], [3, 2], [7, 0]], x)
    check_F(m, x, y, (3, 2))


def test_F_2(for_F):
    m, x, y = for_F
    m += y >= F([[-4, 5], [-2, -1], [-1, 3]], x)
    check_F(m, x, y, (-2, -1))


def test_F_3(for_F):
    m, x, y = for_F
    y.obj = -1
    m += y == F([[1, 3], [5, 2], [9, 8], [11, 6]], x)
    check_F(m, x, y, (9, 8))


def test_F_4(for_F):
    m, x, y = for_F
    m += y == F([[1, 3], [5, 2], [9, 8], [11, 6]], x)
    check_F(m, x, y, (5, 2))


def test_addvars_1():
    df = pd.DataFrame([[], []])
    m = Model()
    v = addvars(m, df)
    assert [i.name for i in v] == ["Var_0", "Var_1"]
    assert [i.var_type for i in v] == ["C", "C"]


def test_addbinvars_1():
    df = pd.DataFrame([[], []])
    m = Model()
    v = addbinvars(m, df)
    assert [i.name for i in v] == ["Var_0", "Var_1"]
    assert [i.var_type for i in v] == ["B", "B"]


def test_addintvars_1():
    df = pd.DataFrame([[], []])
    m = Model()
    v = addintvars(m, df)
    assert [i.name for i in v] == ["Var_0", "Var_1"]
    assert [i.var_type for i in v] == ["I", "I"]


def test_Series_1():
    A = pd.DataFrame([[1, 2], [3, 1]])
    b = pd.Series([16, 18])
    m = Model()
    var = m.add_var_tensor((2,), "var")
    m.objective = maximize(xsum(100 * var))
    m += A @ var <= b
    m.verbose = 0
    m.optimize()
    assert all(np.equal(var.astype(float), [4, 6]))
