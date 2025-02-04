from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys

sys.path.append('../')

import deepxdeNEW as dde


def main():
    def func_lo(x):
        A, B, C = 0.5, 10, -5
        return A * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + B * (x - 0.5) + C

    def func_hi(x):
        return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

    geom = dde.geometry.Interval(0, 1)
    num_test = 1000
    data = dde.data.MfFunc(geom, func_lo, func_hi, 100, 6, num_test)

    activation = "tanh"
    initializer = "Glorot uniform"
    regularization = ["l2", 0.01]
    net = dde.maps.MfNN(
        [1] + [20] * 4 + [1],
        [10] * 2 + [1],
        activation,
        initializer,
        regularization=regularization,
    )

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=80000)

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
