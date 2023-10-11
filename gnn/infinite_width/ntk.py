import math

from neural_tangents import stax


def nngps_and_ntks(layer, x, bias=0, weight=1, nonlinear=True, classes=1):
    bias = math.sqrt(bias) # neural-tangents takes sigma instead of sigma^2
    #nonlinearity = stax.Relu()
    nonlinearity = stax.Sigmoid_like()
    #nonlinearity = stax.Cos()
    #nonlinearity = stax.Erf()
    if nonlinear:
        _, _, kernel_fn1 = [
            stax.serial(stax.Dense(1, b_std=bias, W_std=weight), nonlinearity, stax.Dense(classes, b_std=bias, W_std=weight)),
            stax.serial(
                stax.Dense(1, b_std=bias, W_std=weight),
                nonlinearity,
                stax.Dense(1, b_std=bias, W_std=weight),
                nonlinearity,
                stax.Dense(classes, b_std=bias, W_std=weight),
            ),
            stax.serial(
                stax.Dense(1, b_std=bias, W_std=weight),
                nonlinearity,
                stax.Dense(1, b_std=bias, W_std=weight),
                nonlinearity,
                stax.Dense(1, b_std=bias, W_std=weight),
                nonlinearity,
                stax.Dense(classes, b_std=bias, W_std=weight),
            ),
            stax.serial(
                stax.Dense(1, b_std=bias, W_std=weight),
                nonlinearity,
                stax.Dense(1, b_std=bias, W_std=weight),
                nonlinearity,
                stax.Dense(1, b_std=bias, W_std=weight),
                nonlinearity,
                stax.Dense(1, b_std=bias, W_std=weight),
                nonlinearity,
                stax.Dense(classes, b_std=bias, W_std=weight),
            ),
        ][layer - 2]
    else:
        _, _, kernel_fn1 = stax.serial(stax.Dense(1, b_std=bias, W_std=weight), stax.Dense(classes, b_std=bias, W_std=weight))

    return kernel_fn1(x, None, "nngp"), kernel_fn1(x, None, "ntk")
