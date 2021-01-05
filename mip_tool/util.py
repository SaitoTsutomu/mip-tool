import os
import sys
from tempfile import TemporaryDirectory
from typing import Sequence

import numpy as np
from mip import BINARY, LinExpr, Model, Var, xsum
from more_itertools import pairwise

Point = Sequence[float]


def add_line(m: Model, p1: Point, p2: Point, x: Var, y: Var, under: bool) -> None:
    """Add constraint which pass through p1 and p2.

    :param m: Mpdel
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

    :param m: Mpdel
    :param curve: Point ndarray
    :param x: Var x
    :param y: Var y
    :param upward: Convex upward if True, defaults to False
    """
    for p1, p2 in pairwise(curve):
        add_line(m, p1, p2, x, y, upward)


def add_lines(m: Model, curve: np.ndarray, x: Var, y: Var):
    """Add non-convex piecewise linear constraint

    :param m: Mpdel
    :param curve: Point ndarray
    :param x: Var x
    :param y: Var y
    """
    n = curve.shape[0]
    w = m.add_var_tensor((n - 1,), "w")
    z = m.add_var_tensor((n - 2,), "z", var_type=BINARY)
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
    with TemporaryDirectory() as dir_:
        fnam = os.path.join(dir_, "dummy.lp")
        m.write(fnam)
        with open(fnam) as fp:
            print(fp.read(), file=out)
