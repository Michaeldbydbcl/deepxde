from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys

sys.path.append('../../')

import deepxdeNEW as dde

# This file is built upon the previous 2-levels multi-fidelity model
# THe title of this file "L2H" means one has 2 "L"ow fidelity datasets and 1 "H"igh fidelity dataset


def main():
    # The previous example from DeepXDE repository
    # def func_lo(x):
    #     A, B, C = 0.5, 10, -5
    #     return A * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + B * (x - 0.5) + C

    # def func_hi(x):
    #     return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

    # Define two functions with low fidelities, and one function with high fidelity
    # def func_lo_one(x):
    #     return 3*x

    def func_lo_one(x):
        return 3 * x

    def func_lo_two(x):
        return np.sin(2 * np.pi * x)

    def func_hi(x):
        return 3 * x * np.sin(2 * np.pi * x)

    geom = dde.geometry.Interval(0, 1)
    num_test = 1000
    data = dde.data.MfFunc_L2H(geom, func_lo_one, func_lo_two, func_hi, 400, 400, 100, num_test)

    activation = "tanh"
    initializer = "Glorot uniform"
    regularization = ["l2", 0.01]

    # The mfnn_L2H net contains 2 low fidelity and 1 high fidelity datasets
    # 

    net = dde.maps.MfNN_L2H(
        [1] + [4] * 4 + [1],   ### Two fidelity layers
        [1] + [4] * 4 + [1],   ### Two fidelity layers
        [2] * 2 + [1],
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
