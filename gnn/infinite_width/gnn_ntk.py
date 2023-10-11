import numpy as np

from gnn.infinite_width.nonlinearities import Sin
from gnn.infinite_width.nonlinearities_numpy import ABRelu, Erf, Sigmoid_like, Cos


def gnn_kernel(adj, x, layers, b=0, w=1, nonlinear_=True):
    adj = adj.astype(np.float64)
    x = x.astype(np.float64)

    if nonlinear_:
        nonlin = ABRelu(0, 1)
        #nonlin = Cos()
        #nonlin = Sin()
        #nonlin = Erf()
    else:
        nonlin = lambda x: (x, np.ones_like(x))

    def forward(adj, nngp, ntk, counter):
        if counter == layers:
            return nngp, ntk
        nngp, sigma_dot = nonlin(nngp)
        ntk = nngp + (ntk * sigma_dot)
        nngp = w * (adj @ nngp @ adj.T) + b
        ntk = w * (adj @ ntk @ adj.T) + b
        return forward(adj, nngp, ntk, counter + 1)

    kernel_init = 1 / x.shape[1] * x @ x.T
    kernel_init = adj @ kernel_init @ adj.T
    kernel_init = w * kernel_init + b
    return forward(adj, kernel_init, kernel_init, 1)


def skip_gnn_kernel(adj, x, layers, b=0, w=1, nonlinear=True):
    if nonlinear:
        nonlinear = ABRelu(0, 1)
        #nonlin = Cos()
        #nonlin = Sin()
        #nonlin = Erf()
    else:
        nonlinear = lambda x: (x, np.ones_like(x))

    def forward(adj, nngp, ntk, counter):
        if counter == layers:
            return nngp, ntk
        else:
            nngp_new, ntk_ = nonlinear(nngp)
            nngp_new = 0.5 * (nngp_new + nngp)
            ntk_ = 0.5 * (ntk + 1)
            ntk = nngp_new + (ntk * ntk_)
            ntk = w * adj @ ntk @ adj.T
            nngp = w * adj @ nngp_new @ adj.T
        return forward(adj, nngp + b, ntk + b, counter + 1)

    kernel_init = 1 / x.shape[1] * x @ x.T
    kernel_init = adj @ kernel_init @ adj.T
    kernel_init = w * kernel_init + b
    nngp, sigma_dot = ABRelu(0, 1)(kernel_init, True)
    ntk = nngp + (sigma_dot * sigma_dot)
    nngp = adj @ nngp @ adj.T
    ntk = adj @ ntk @ adj.T
    return forward(adj, w * nngp + b, w * ntk + b, 2)
