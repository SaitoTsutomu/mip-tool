import sys

from IPython.display import HTML
from mip import ParameterNotAvailable

STYLE = ' style="text-align: left;" width="64"'
VAR_WIDTH = 64


def main():
    src = f"{sys.argv[1]};" if len(sys.argv) > 1 else ""
    exec(f"import mip_tool.view;from mip import *;m = Model(solver_name='CBC');{src}print(view_model(m).data)")


def var_desc(v):
    lb, ub = v.lb, v.ub
    if v.var_type == "C":
        s = "連続変数" if lb else "非負変数"
    elif v.var_type == "B":
        s = "0-1変数"
        ub = float("inf")
    elif v.var_type == "I":
        s = "整数変数" if lb else "非負整数変数"
    if lb and lb > -1e308:  # noqa: PLR2004
        s += f", ≧{lb}"
    if ub < 1e308:  # noqa: PLR2004
        s += f", ≦{ub}"
    return s


def view_var(vs):
    lst = [f"<tr><td{STYLE}>変数</td><td>なし</td></tr>"]
    if vs:
        lst = []
        for v in vs:
            lst.append(f'<tr><td width="{VAR_WIDTH}">{v.name} :</td><td>{var_desc(v)}</td></tr>\n')
        lst[0] = f'<tr><td rowspan="{len(lst)}"{STYLE}>変数</td>' + lst[0][4:]
    return f'<table width="100%">\n{"".join(lst)}</table>'


def view_obj(sense, obj):
    s = "なし"
    if obj:
        s = f"{obj} → 最{'小' if sense == 'MIN' else '大'}化"
        s = s.removeprefix("+ ")
    return f'<table width="100%"><tr><td{STYLE}>目的関数</td><td>{s}</td></tr></table>'


def view_const(constrs):
    lst = [f"<tr><td{STYLE}>制約条件</td><td>なし</td></tr>"]
    if constrs:
        lst = []
        for constr in constrs:
            s = str(constr.expr).removeprefix("+ ")
            s = s.replace("<=", "≦").replace(">=", "≧")
            lst.append(f"<tr><td>{s}</td></tr>\n")
        lst[0] = f'<tr><td rowspan="{len(lst)}"{STYLE}>制約条件</td>' + lst[0][4:]
    return f'<table width="100%">\n{"".join(lst)}</table>'


def view_model(m, width="320", *, has_html=False):
    try:
        obj = m.objective
    except ParameterNotAvailable:
        obj = None
    s = f"""\
<table width="{width}">
  <tr><td style="text-align: center;">モデル</td></tr>
  <tr><td>{view_var(m.vars)}</td></tr>
  <tr><td>{view_obj(m.sense, obj)}</td></tr>
  <tr><td>{view_const(m.constrs)}</td></tr>
</table>"""
    if has_html:
        s = f"<html>\n{s}\n</html>"
    return HTML(s)
