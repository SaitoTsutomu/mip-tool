# MIP-Tool

MIP-Tool is a package for [Python-MIP](https://www.python-mip.com/).

## Installation

```
pip install mip-tool
```

## Example

## Show/View model

```python
from mip import Model, maximize
from mip_tool import show_model

m = Model()
x = m.add_var("x")
y = m.add_var("y")
m.objective = maximize(x + y)
m += x + 2 * y <= 16
m += 3 * x + y <= 18
show_model(m)
```

*Output*

```
\Problem name: 

Minimize
OBJROW: - x - y
Subject To
constr(0):  x + 2 y <= 16
constr(1):  3 x + y <= 18
Bounds
End
```

In Jupyter

```python
from mip_tool.view import view_model

view_model(m)
```

*Output*

<table width="320">
  <tr><td style="text-align: center;">モデル</td></tr>
  <tr><td><table width="100%">
<tr><td rowspan="2" style="text-align: left;" width="64">変数</td><td width="64">x :</td><td>非負変数</td></tr>
<tr><td width="64">y :</td><td>非負変数</td></tr>
</table></td></tr>
  <tr><td><table width="100%"><tr><td style="text-align: left;" width="64">目的関数</td><td>x + y  → 最大化</td></tr></table></td></tr>
  <tr><td><table width="100%">
<tr><td rowspan="2" style="text-align: left;" width="64">制約条件</td><td>x + 2.0y  ≦ 16.0</td></tr>
<tr><td>3.0x + y  ≦ 18.0</td></tr>
</table></td></tr>
</table>

### Non-convex piecewise linear constraint

Maximize y which is on points of (-2, 6), (-1, 7), (2, -2), (4, 5).

```python
import numpy as np
from mip import INF, Model, OptimizationStatus
from mip_tool import add_lines, show_model

m = Model(solver_name="CBC")
x = m.add_var("x", lb=-INF)
y = m.add_var("y", obj=-1)
curve = np.array([[-2, 6], [-1, 7], [2, -2], [4, 5]])
add_lines(m, curve, x, y)
m.verbose = 0
m.optimize()
assert m.status == OptimizationStatus.OPTIMAL
assert (x.x, y.x) == (-1, 7)
show_model(m)
```

*Output*

```
\Problem name: 

Minimize
OBJROW: - y
Subject To
constr(0):  x - w_0 - w_1 - w_2 = -2
constr(1):  y - w_0 + 3 w_1 -3.50000 w_2 = 6
constr(2):  - w_0 + z_0 <= -0
constr(3):  w_0 <= 1
constr(4):  - w_1 + 3 z_1 <= -0
constr(5):  w_1 -3 z_0 <= -0
constr(6):  w_2 -2 z_1 <= -0
Bounds
 x Free
 0 <= z_0 <= 1
 0 <= z_1 <= 1
Integers
z_0 z_1 
End
```

## F example

Easy to understand using F.

attention: Change Model and Var when using mip_tool.func.

```python
from mip_tool.func import F

m = Model(solver_name="CBC")
x = m.add_var("x")
y = m.add_var("y", obj=-1)
m += y <= F([[0, 2], [1, 3], [2, 2]], x)
m.verbose = 0
m.optimize()
print(x.x, y.x)  # 1.0 3.0
```

- `y <= F(curve, x)` and `y >= F(curve, x)` call `add_lines_conv`.
- `y == F(curve, x)` calls `add_lines`.


## pandas.DataFrame example

attention: Change Series when using mip_tool.func.

```python
import pandas as pd
from mip import Model, maximize, xsum
from mip_tool.func import addvars

A = pd.DataFrame([[1, 2], [3, 1]])
b = pd.Series([16, 18])
m = Model(solver_name="CBC")
x = addvars(m, A, "X", False)
m.objective = maximize(xsum(x))
m += A @ x <= b
m.verbose = 0
m.optimize()
print(x.astype(float))  # [4. 6.]
```

Expression `m += A.T.apply(lambda row: xsum(row * x)) <= b` may be faster than `m += A @ x <= b`.
