"""
To plot and study kernel for c4der
"""


import numpy as np
import plotly.graph_objects as go


def f(x, y, eps1=2.5, eps2=3.5, ar=0.6, lambd=0.3):
    Z = np.sqrt(lambd * (x**2) + (1 - lambd) * (y**2)) - eps1
    Z[Z < 0] = 0.0
    Z[Z - (eps2 - eps1) > 0] = np.inf
    sigma = ((eps1 - eps2) ** 2) / np.log(ar)
    return np.exp((Z**2) / sigma)


def f1(x, y, eps=0, sigma=1000):
    Z = np.sqrt(x**2 + y**2) - eps
    Z[Z < 0] = 0.0
    return (1 - (Z / sigma) ** 3) ** 3


x = np.linspace(-20, 20, 100)
y = np.linspace(-20, 20, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = go.Figure(go.Surface(x=X, y=Y, z=Z))
fig.show()
