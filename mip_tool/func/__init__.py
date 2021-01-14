import numpy as np

from .. import add_lines, add_lines_conv

if "_org_iadd" not in globals():
    from mip import Model, Var

    _org_iadd = Model.__iadd__
    _org_eq, _org_le, _org_ge = Var.__eq__, Var.__le__, Var.__ge__

    def _new_iadd(self, other):
        if isinstance(other, F):
            return other._iadd(self)
        return _org_iadd(self, other)

    def _new_eq(self, other):
        try:
            return _org_eq(self, other)
        except TypeError:
            return NotImplemented

    def _new_le(self, other):
        try:
            return _org_le(self, other)
        except TypeError:
            return NotImplemented

    def _new_ge(self, other):
        try:
            return _org_ge(self, other)
        except TypeError:
            return NotImplemented

    Model.__iadd__ = _new_iadd
    Var.__eq__, Var.__le__, Var.__ge__ = _new_eq, _new_le, _new_ge


class F:
    """`y <= F(curve, x)`, curve is piecewise linear points"""

    def __init__(self, curve, x):
        self.curve, self.x = curve, x
        if not isinstance(self.curve, np.ndarray):
            self.curve = np.array(self.curve)

    def __eq__(self, y):
        self.y, self.sense = y, "="
        return self

    def __le__(self, y):
        self.y, self.sense = y, ">"
        return self

    def __ge__(self, y):
        self.y, self.sense = y, "<"
        return self

    def _iadd(self, m):
        assert isinstance(self.y, Var)
        if self.sense == "=":
            add_lines(m, self.curve, self.x, self.y)
        else:
            add_lines_conv(m, self.curve, self.x, self.y, self.sense == "<")
        return m
