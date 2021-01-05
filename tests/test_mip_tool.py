import numpy as np
from mip import INF, Model, OptimizationStatus
from mip_tool import add_line, add_lines, add_lines_conv


def check_add_line(xargs, yargs, p1, p2, under, expect):
    """1制約条件の確認"""
    m = Model()
    x = m.add_var("x", **xargs)
    y = m.add_var("y", **yargs)
    add_line(m, p1, p2, x, y, under)
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
