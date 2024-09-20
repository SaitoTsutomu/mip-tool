import numpy as np

from .. import add_lines, add_lines_conv

if "_mdl_iadd" not in globals():
    from mip import LinExpr, LinExprTensor, Model, Var, xsum
    from pandas import Series
    from pandas.core.groupby import SeriesGroupBy

    _mdl_iadd = Model.__iadd__
    _var_eq, _var_le, _var_ge = Var.__eq__, Var.__le__, Var.__ge__
    _ser_eq, _ser_le, _ser_ge = Series.__eq__, Series.__le__, Series.__ge__

    def _mdl_iadd_n(self, other):
        if isinstance(other, F):
            return other._iadd(self)  # noqa: SLF001
        if isinstance(other, Series):
            return _mdl_iadd(self, other.values.view(LinExprTensor))  # noqa: PD011
        return _mdl_iadd(self, other)

    def _var_eq_n(self, other):
        try:
            return _var_eq(self, other)
        except TypeError:
            return NotImplemented

    def _var_le_n(self, other):
        try:
            return _var_le(self, other)
        except TypeError:
            return NotImplemented

    def _var_ge_n(self, other):
        try:
            return _var_ge(self, other)
        except TypeError:
            return NotImplemented

    def _ser_eq_n(self, other):
        if self.size and isinstance(self.iloc[0], LinExpr):
            return self.values.view(LinExprTensor) == other
        return _ser_eq(self, other)

    def _ser_le_n(self, other):
        if self.size and isinstance(self.iloc[0], LinExpr):
            return self.values.view(LinExprTensor) <= other
        return _ser_le(self, other)

    def _ser_ge_n(self, other):
        if self.size and isinstance(self.iloc[0], LinExpr):
            return self.values.view(LinExprTensor) >= other
        return _ser_ge(self, other)

    def _sgb_xsum(self):
        return self.apply(xsum).values.view(LinExprTensor)  # noqa: PD011

    Model.__iadd__ = _mdl_iadd_n
    Var.__eq__, Var.__le__, Var.__ge__ = _var_eq_n, _var_le_n, _var_ge_n
    Series.__eq__, Series.__le__, Series.__ge__ = _ser_eq_n, _ser_le_n, _ser_ge_n
    SeriesGroupBy.xsum = _sgb_xsum


class F:  # noqa: PLW1641
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
            add_lines_conv(m, self.curve, self.x, self.y, upward=self.sense == "<")
        return m


def addvars(m, df, name="Var", add=True, **kwargs):  # noqa: FBT002
    v = m.add_var_tensor((len(df),), name=name, **kwargs)
    if add:
        df[name] = v
    return v


def addbinvars(m, df, **kwargs):
    return addvars(m, df, var_type="B", **kwargs)


def addintvars(m, df, **kwargs):
    return addvars(m, df, var_type="I", **kwargs)
