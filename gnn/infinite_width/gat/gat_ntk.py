import numpy as np

from gnn.infinite_width.gat.utils import ia_x_ia_T, batch_mul, torch_matmul
from gnn.infinite_width.nonlinearities_numpy import ABRelu, Erf


def gat_ntk(adj, x, layers, b=0, w=1, nonlinear=False):
    # From the paper: \sigma_{2} is relu (or something else) whereas \sigma_{1} is the Identity (only one nonlinearity)
    if nonlinear:
        nonlin = ABRelu(0.2, 1)
        #nonlin = Cos()
        #nonlin = Sin()
        #nonlin = Erf()
    else:
        nonlin = lambda x: (x, np.ones_like(x))
    ia = ia_x_ia_T(adj)

    def forward(nngp, ntk, counter):
        if counter == layers:
            return nngp, ntk
        nngp, ntk = w * nngp + b, w * ntk + b
        sigma, sigma_dot = nonlin(nngp)
        cov_att = ia(sigma)
        nngp = batch_mul(sigma, cov_att)
        helper = ntk * sigma_dot
        ntk = batch_mul(sigma, ia(helper))
        ntk = ntk + batch_mul(helper, cov_att)
        ntk = 2 * nngp + ntk
        return forward(nngp, ntk, counter + 1)

    kernel_init = 1 / x.shape[1] * x @ x.T
    return forward(kernel_init, kernel_init, 1)


def gat_ntk2(adj, x, layers, b=0, w=1):
    # Two Nonlinearities
    nonlin1 = ABRelu(0.2, 1)
    #nonlin2 = Cos()
    nonlin2 = Erf()
    dtype = np.float32
    ia = ia_x_ia_T(adj, dtype=dtype)

    def forward(nngp, ntk, counter):
        if counter == layers:
            return nngp, ntk
        nngp, ntk = w * nngp + b, w * ntk + b
        sigma, sigma_dot = nonlin1(nngp)  # corresponds to second nonlinearity sigma_{2}
        _sigma_2, _sigma_2_dot = nonlin2(nngp)  # corresponds to second nonlinearity sigma_{1}
        phi = ia(sigma)
        helper = sigma_dot * ntk
        nngp = batch_mul(sigma, phi)
        ntk = nngp + batch_mul(helper, phi)  # first and last term of equation
        del phi
        phi_dot = ia(_sigma_2_dot)
        ntk = ntk + batch_mul(sigma, torch_matmul(ia(sigma), phi_dot))  # second term of equation
        ntk = ntk + batch_mul(sigma, torch_matmul(ia(helper), phi_dot))  # third term of equation
        return nngp, ntk

    kernel_init = 1 / x.shape[1] * x @ x.T
    kernel_init = kernel_init.astype(dtype)
    return forward(kernel_init, kernel_init, 1)


def gat_ntk3(adj, x, layers, b=0, w=1):
    # Two Nonlinearities
    nonlin1 = ABRelu(0.2, 1)
    # nonlin2 = Sin()
    nonlin2 = Erf()
    dtype = np.float32
    ia = ia_x_ia_T(adj, dtype=dtype)

    def forward(nngp, ntk, counter):
        if counter == layers:
            return nngp, ntk
        nngp, ntk = w * nngp + b, w * ntk + b
        sigma, sigma_dot = nonlin1(nngp)  # corresponds to second nonlinearity sigma_{2}
        _sigma_2, _sigma_2_dot = nonlin2(sigma)  # corresponds to second nonlinearity sigma_{1}
        phi = ia(sigma)
        helper = sigma_dot * ntk
        nngp = batch_mul(sigma, phi)
        ntk = nngp + batch_mul(helper, phi)  # first and last term of equation
        del phi
        phi_dot = ia(_sigma_2_dot)
        ntk = ntk + batch_mul(sigma, torch_matmul(ia(sigma), phi_dot))  # second term of equation
        ntk = ntk + batch_mul(sigma, torch_matmul(ia(helper), phi_dot))  # third term of equation
        return nngp, ntk

    kernel_init = 1 / x.shape[1] * x @ x.T
    kernel_init = kernel_init.astype(dtype)
    return forward(kernel_init, kernel_init, 1)

def gat_ntk4(adj, x, layers, b=0, w=1):
    # Two Nonlinearities
    nonlin1 = ABRelu(0.2, 1)
    nonlin1 = Erf(c=1)
    # nonlin2 = Sin()
    nonlin2 = Erf(c=1)
    dtype = np.float32
    ia = ia_x_ia_T(adj, dtype=dtype)

    def forward(nngp, ntk, counter):
        if counter == layers:
            return nngp, ntk
        nngp, ntk = w * nngp + b, w * ntk + b
        sigma, sigma_dot = nonlin1(nngp)  # corresponds to second nonlinearity sigma_{2}
        _sigma_2, _sigma_2_dot = nonlin2(sigma)  # corresponds to second nonlinearity sigma_{1}
        phi = ia(sigma)
        helper = sigma_dot * ntk
        nngp = batch_mul(sigma, phi)
        ntk = nngp + batch_mul(helper, phi)  # first and last term of equation
        del phi
        phi_dot = ia(_sigma_2_dot)
        ntk = ntk + batch_mul(sigma, torch_matmul(ia(sigma), phi_dot))  # second term of equation
        ntk = ntk + batch_mul(sigma, torch_matmul(ia(helper), phi_dot))  # third term of equation
        return nngp, ntk

    kernel_init = 1 / x.shape[1] * x @ x.T
    kernel_init = kernel_init.astype(dtype)
    return forward(kernel_init, kernel_init, 1)