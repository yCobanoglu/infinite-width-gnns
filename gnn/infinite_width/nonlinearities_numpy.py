import scipy
import numpy as np
import functools
from typing import Optional, Tuple
import numpy as np
import numpy as onp
import numpy as jnp


def Erf(a: float = 1.0, b: float = 1.0, c: float = 0.0):
    if b != 1.0:
        raise NotImplementedError("Something wrong for Erf checked against Neural Tangents")
    def _fn(nngp):
        cov = nngp * b
        cov = jnp.diag(cov)
        cov1_denom = 1 + 2 * cov
        prod = jnp.outer(cov1_denom, cov1_denom)
        factor = 2 / np.pi

        square_root = _sqrt(prod - 4 * nngp**2)
        nngp = factor * np.arctan2(2 * nngp, square_root)
        dot_sigma = 2 * factor / square_root
        return a * nngp + c, a * dot_sigma + c

    return _fn


def Sigmoid_like():
    """A sigmoid like function `f(x) = .5 * erf(x / 2.4020563531719796) + .5`.

    The constant `2.4020563531719796` is chosen so that the squared loss between
    this function and the ground truth sigmoid is minimized on the interval
    `[-5, 5]`; see
    https://gist.github.com/SiuMath/679e8bb4bce13d5f2383a27eca649575.

    Returns:
      `(init_fn, apply_fn, kernel_fn)`.
    """
    return Erf(a=0.5, b=1 / 2.4020563531719796, c=0.5)


def _sqrt(x, tol=0.0):
    return np.sqrt(np.maximum(x, tol))


def Monomial(degree: int):
    def kernel_fn(nngp, ntk=None):
        prod12 = scipy.sparse.csr_array.diagonal(nngp)
        prod12 = scipy.sparse.csr_array(prod12)

        def nngp_ntk_fn(nngp: np.ndarray, prod: np.ndarray, ntk: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            def nngp_fn(nngp: np.ndarray, degree: int) -> np.ndarray:
                if degree == -1:
                    nngp = np.zeros_like(nngp)

                elif degree == 0:
                    nngp = np.ones_like(nngp)

                elif degree == 1:
                    pass

                elif degree == 2:
                    nngp = 2 * nngp**2 + prod

                elif degree == 3:
                    nngp = 6 * nngp**3 + 9 * nngp * prod

                elif degree == 4:
                    nngp = 3 * (8 * nngp**4 + 3 * prod * (8 * nngp**2 + prod))

                elif degree == 5:
                    nngp = 15 * nngp * (8 * nngp**4 + 5 * prod * (8 * nngp**2 + 3 * prod))

                else:
                    raise NotImplementedError(degree)

                return nngp

            if ntk is not None:
                ntk *= degree**2 * nngp_fn(nngp, degree - 1)

            nngp = nngp_fn(nngp, degree)
            return nngp, ntk

        nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)
        return nngp, ntk

    return kernel_fn


def _arctan2(x, y, fill_zero: Optional[float] = None):
    if fill_zero is not None:
        return np.where(np.bitwise_and(x == 0.0, y == 0.0), fill_zero, np.arctan2(x, y))
    return np.arctan2(x, y)


def ABRelu(a: float, b: float, do_stabilize: bool = False):
    def _fn(nngp, compute_sigma_dot=True):
        cov = jnp.diag(nngp)

        if do_stabilize:
            factor = jnp.maximum(jnp.max(jnp.abs(nngp)), 1e-12)
            nngp /= factor
            cov /= factor

        prod = jnp.outer(cov, cov)

        square_root = _sqrt(prod - nngp**2)
        angles = _arctan2(square_root, nngp, fill_zero=jnp.pi / 2)

        factor = (a - b) ** 2 / (2 * jnp.pi)
        dot_sigma = (a**2 + b**2) / 2 - factor * angles
        nngp = factor * square_root + dot_sigma * nngp

        if do_stabilize:
            nngp *= factor
            cov *= factor

        if compute_sigma_dot:
            return nngp, dot_sigma
        return nngp

    return _fn


def Sin(a: float = 1.0, b: float = 1.0, c: float = 0.0):
    """Affine transform of `Sin` nonlinearity, i.e. `a sin(b*x + c)`.

    Args:
      a: output scale.
      b: input scale.
      c: input phase shift.
    Returns:
      `(init_fn, apply_fn, kernel_fn)`.
    """

    def kernel_fn(nngp, ntk=True):
        cov = np.diag(nngp)

        sum = onp.add.outer(cov, cov)
        # sum=  utils.outer_prod(cov, cov, 0, 1, op.add)

        half_a_square = a**2 / 2.0

        def nngp_ntk_fn(nngp, sum_, ntk=True):
            s1 = np.exp(b**2 * (-0.5 * sum_ + nngp))
            s2 = np.exp(b**2 * (-0.5 * sum_ - nngp)) * np.cos(2 * c)
            nngp = half_a_square * (s1 - s2)
            if ntk:
                ntk = half_a_square * b**2 * (s1 + s2)
            return nngp, ntk

        return nngp_ntk_fn(nngp, sum, ntk)

    return kernel_fn


def Cos(a: float = 1.0, b: float = 1.0, c: float = 0.0):
    """Affine transform of `Cos` nonlinearity, i.e. `a cos(b*x + c)`.

    Args:
      a: output scale.
      b: input scale.
      c: input phase shift.
    Returns:
      `(init_fn, apply_fn, kernel_fn)`.
    """
    return Sin(a=a, b=b, c=c + np.pi / 2)
