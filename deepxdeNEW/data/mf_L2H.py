from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import preprocessing

from .data import Data
from ..backend import tf
from ..utils import run_if_any_none

class MfFunc_L2H(Data):
    """Multifidelity function approximation.
        Data with two low fidelity and one high fidelity functions, 
        which will be fed to the L2H Multi-fidelity NN. 
    """

    # def __init__(
    #     self, geom, func_lo, func_hi, num_lo, num_hi, num_test, dist_train="uniform"
    # ):

    def __init__(
        self, geom, func_lo_one, func_lo_two, func_hi, num_lo_one, num_lo_two, num_hi, num_test, dist_train="uniform"
    ):

        self.geom = geom
        # self.func_lo = func_lo

        self.func_lo_one = func_lo_one   ### Two fidelity datasets
        self.func_lo_two = func_lo_two   ### Two fidelity datasets

        self.func_hi = func_hi

        # self.num_lo = num_lo
        self.num_lo_one = num_lo_one     ### Two fidelity datasets
        self.num_lo_two = num_lo_two     ### Two fidelity datasets

        self.num_hi = num_hi
        self.num_test = num_test
        self.dist_train = dist_train

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def losses(self, targets, outputs, loss, model):
        # loss_lo = loss(targets[0][: self.num_lo], outputs[0][: self.num_lo])

        loss_lo_one = loss(targets[0][: self.num_lo_one], outputs[0][: self.num_lo_one])    ### Two fidelity layers
        loss_lo_two = loss(targets[1][self.num_lo_one: self.num_lo_one + self.num_lo_two], outputs[1][self.num_lo_one: self.num_lo_one + self.num_lo_two])    ### Two fidelity layers

        # print("This is what we want to see:")
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(targets[1][self.num_lo_one: self.num_lo_one + self.num_lo_two].shape)
        # print(self.num_lo_one, self.num_lo_one + self.num_lo_two)

        loss_hi = loss(targets[2][self.num_lo_one + self.num_lo_two :], outputs[2][self.num_lo_one + self.num_lo_two :])   #### !!!!! what about two?

        return [loss_lo_one, loss_lo_two, loss_hi]

    @run_if_any_none("X_train", "y_train")
    def train_next_batch(self, batch_size=None):
        if self.dist_train == "uniform":
            self.X_train = np.vstack(
                (
                    # self.geom.uniform_points(self.num_lo, True),

                    self.geom.uniform_points(self.num_lo_one, True),   ### Two low fidelity layers
                    self.geom.uniform_points(self.num_lo_two, True),   ### Two low fidelity layers

                    self.geom.uniform_points(self.num_hi, True),
                )
            )
        else:
            self.X_train = np.vstack(
                (
                    # self.geom.random_points(self.num_lo, "sobol"),

                    self.geom.random_points(self.num_lo_one, "sobol"),   ### Two low fidelity layers
                    self.geom.random_points(self.num_lo_two, "sobol"),   ### Two low fidelity layers

                    self.geom.random_points(self.num_hi, "sobol"),
                )
            )
        # y_lo_train = self.func_lo(self.X_train)

        y_lo_one_train = self.func_lo_one(self.X_train)   ### Two low fidelity layers
        y_lo_two_train = self.func_lo_two(self.X_train)   ### Two low fidelity layers

        y_hi_train = self.func_hi(self.X_train)
        self.y_train = [y_lo_one_train, y_lo_two_train, y_hi_train]
        return self.X_train, self.y_train

    @run_if_any_none("X_test", "y_test")
    def test(self):
        self.X_test = self.geom.uniform_points(self.num_test, True)
        # y_lo_test = self.func_lo(self.X_test)

        y_lo_one_test = self.func_lo_one(self.X_test)    ### Two low fidelity layers
        y_lo_two_test = self.func_lo_two(self.X_test)    ### Two low fidelity layers

        y_hi_test = self.func_hi(self.X_test)
        self.y_test = [y_lo_one_test, y_lo_two_test, y_hi_test]
        return self.X_test, self.y_test



class MfData_L2H(Data):
    """Multifidelity function approximation from data set.
        Data with two low fidelity and one high fidelity datasets, 
        which will be fed to the L2H Multi-fidelity NN. 

    Args:
        col_x: List of integers.
        col_y: List of integers.
    """

    def __init__(
        self,
        # X_lo_train=None,
        X_lo_one_train=None,      ##### Add two low fidelity datasets
        X_lo_two_train=None,      ##### Add two low fidelity datasets

        X_hi_train=None,

        # y_lo_train=None,
        y_lo_one_train=None,      #### Add two low fidelity datasets
        y_lo_two_train=None,      #### Add two low fidelity datasets

        y_hi_train=None,
        X_hi_test=None,
        y_hi_test=None,

        # fname_lo_train=None,
        fname_lo_one_train=None,    ##### Add two low fidelity datasets
        fname_lo_two_train=None,    ##### Add two low fidelity datasets

        fname_hi_train=None,
        fname_hi_test=None,
        col_x=None,
        col_y=None,
    ):
        if X_lo_one_train is not None:       ##### What about two?
            # self.X_lo_train = X_lo_train
            self.X_lo_one_train = X_lo_one_train      ##### Add two low fidelity datasets
            self.X_lo_two_train = X_lo_two_train      ##### Add two low fidelity datasets

            self.X_hi_train = X_hi_train
            # self.y_lo_train = y_lo_train
            self.y_lo_one_train = y_lo_one_train          ##### Add two low fidelity datasets
            self.y_lo_two_train = y_lo_two_train          ##### Add two low fidelity datasets

            self.y_hi_train = y_hi_train
            self.X_hi_test = X_hi_test
            self.y_hi_test = y_hi_test
        elif fname_lo_one_train is not None:
            data = np.loadtxt(fname_lo_one_train)     ##### Add two low fidelity datasets
            self.X_lo_one_train = data[:, col_x]      ##### Add two low fidelity datasets
            self.y_lo_one_train = data[:, col_y]      ##### Add two low fidelity datasets
           
            data = np.loadtxt(fname_lo_two_train)     ##### Add two lwo fidelity datasets
            self.X_lo_two_train = data[:, col_x]      ##### Add two lwo fidelity datasets
            self.y_lo_two_train = data[:, col_y]      ##### Add two lwo fidelity datasets

            data = np.loadtxt(fname_hi_train)
            self.X_hi_train = data[:, col_x]
            self.y_hi_train = data[:, col_y]
            data = np.loadtxt(fname_hi_test)
            self.X_hi_test = data[:, col_x]
            self.y_hi_test = data[:, col_y]
        else:
            raise ValueError("No training data.")

        self.X_train = None
        self.y_train = None
        self.scaler_x = None
        self._standardize()

    def losses(self, targets, outputs, loss, model):
        # n1 = tf.cond(model.net.training, lambda: len(self.X_lo_one_train), lambda: 0)
        # n2 = tf.cond(model.net.training, lambda: len(self.X_lo_two_train), lambda: 0)
        n1 = len(self.X_lo_one_train)
        n2 = len(self.X_lo_two_train)

        print("########################################")
        print("n1 is: ", n1)
        print("n2 is: ", n2)

        # loss_lo = loss(targets[0][:n], outputs[0][:n])
        loss_lo_one = loss(targets[0][: n1], outputs[0][: n1])            ##### Add two fidelity datasets
        loss_lo_two = loss(targets[1][n1: n1 + n2], outputs[1][n1: n1 + n2])      ##### Add two fidelity datasets

        loss_hi = loss(targets[2][n1 + n2:], outputs[2][n1 + n2:])
        return [loss_lo_one, loss_lo_two, loss_hi]

    @run_if_any_none("X_train", "y_train")
    def train_next_batch(self, batch_size=None):
        self.X_train = np.vstack((self.X_lo_one_train, self.X_lo_two_train, self.X_hi_train))

        self.y_lo_one_train, self.y_lo_two_train, self.y_hi_train = (
            np.vstack((self.y_lo_one_train, np.zeros_like(self.y_lo_two_train), np.zeros_like(self.y_hi_train))),   ##### Add
            np.vstack((np.zeros_like(self.y_lo_one_train), self.y_lo_two_train, np.zeros_like(self.y_hi_train))),   ##### Add
            np.vstack((np.zeros_like(self.y_lo_one_train), np.zeros_like(self.y_lo_two_train), self.y_hi_train)),   ##### Add
        )
        self.y_train = [self.y_lo_one_train, self.y_lo_two_train, self.y_hi_train]
        return self.X_train, self.y_train

    def test(self):
        return self.X_hi_test, [self.y_hi_test, self.y_hi_test, self.y_hi_test]

    def _standardize(self):
        self.scaler_x = preprocessing.StandardScaler(with_mean=True, with_std=True)
        # self.X_lo_train = self.scaler_x.fit_transform(self.X_lo_train)
        self.X_lo_one_train = self.scaler_x.fit_transform(self.X_lo_one_train)     ##### Add two fidelity datasets
        self.X_lo_two_train = self.scaler_x.fit_transform(self.X_lo_two_train)     ##### Add two fidelity datasets

        self.X_hi_train = self.scaler_x.transform(self.X_hi_train)
        self.X_hi_test = self.scaler_x.transform(self.X_hi_test)


# class DataMF(Data):
#     """Multifidelity function approximation with uncertainty quantification (legacy version).
#     """

#     def __init__(self, flow, fhi, geom):
#         self.flow, self.fhi = flow, fhi
#         self.geom = geom

#         self.train_x, self.train_y = None, None
#         self.test_x, self.test_y = None, None

#     def train_next_batch(self, batch_size):
#         keeps = [0, 2, 5, 8, 10]
#         x = self.geom.uniform_points(batch_size, True)
#         self.train_x = np.empty((0, 1))
#         self.train_y = np.empty((0, 2))
#         for _ in range(10):
#             ylow = self.flow(x)
#             yhi = self.fhi(x)
#             for i in range(batch_size):
#                 if i not in keeps:
#                     yhi[i, 0] = ylow[i, 0] + 2*np.random.randn()
#             self.train_x = np.vstack((self.train_x, x))
#             self.train_y = np.vstack((self.train_y, np.hstack((ylow, yhi))))

#         x = x[keeps]
#         ylow = self.flow(x)
#         yhi = self.fhi(x)
#         for _ in range(500):
#             self.train_x = np.vstack((self.train_x, x))
#             self.train_y = np.vstack((self.train_y, np.hstack((ylow, yhi))))

#         return self.train_x, self.train_y

#     @run_if_any_none('test_x', 'test_y')
#     def test(self, n):
#         self.test_x = self.geom.uniform_points(n, True)
#         ylow = self.flow(self.test_x)
#         yhi = self.fhi(self.test_x)
#         self.test_y = np.hstack((ylow, yhi))
#         return self.test_x, self.test_y
