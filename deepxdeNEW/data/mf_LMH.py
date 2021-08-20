from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import preprocessing

from .data import Data
from ..backend import tf
from ..utils import run_if_any_none

class MfFunc_LMH(Data):
    """Multifidelity function approximation.
        Data with one low fidelity, one middle fidelity and one high fidelity functions, 
        which will be fed to the LMH Multi-fidelity NN. 
    """

    # def __init__(
    #     self, geom, func_lo, func_hi, num_lo, num_hi, num_test, dist_train="uniform"
    # ):

    def __init__(
        self, geom, func_lo, func_mi, func_hi, num_lo, num_mi, num_hi, num_test, dist_train="uniform"
    ):

        self.geom = geom
        self.func_lo = func_lo

        self.func_mi = func_mi   ### Middle fidelity datasets
        self.func_hi = func_hi   ### High fidelity datasets


        self.num_lo = num_lo

        self.num_mi = num_mi     ### Middle fidelity datasets
        self.num_hi = num_hi     ### High fidelity datasets

        self.num_test = num_test
        self.dist_train = dist_train

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def losses(self, targets, outputs, loss, model):
        loss_lo = loss(targets[0][: self.num_lo], outputs[0][: self.num_lo])

        loss_mi = loss(targets[1][self.num_lo: self.num_lo + self.num_mi], outputs[1][self.num_lo: self.num_lo + self.num_mi])    ### Middle fidelity layers
        loss_hi = loss(targets[2][self.num_lo + self.num_mi: ], outputs[2][self.num_lo + self.num_mi: ])    ### High fidelity layers

        return [loss_lo, loss_mi, loss_hi]

    @run_if_any_none("X_train", "y_train")
    def train_next_batch(self, batch_size=None):
        if self.dist_train == "uniform":
            self.X_train = np.vstack(
                (
                    self.geom.uniform_points(self.num_lo, True),

                    self.geom.uniform_points(self.num_mi, True),   ### Middle fidelity layers
                    self.geom.uniform_points(self.num_hi, True),   ### HIGH fidelity layers

                )
            )
        else:
            self.X_train = np.vstack(
                (
                    self.geom.random_points(self.num_lo, "sobol"),

                    self.geom.random_points(self.num_mi, "sobol"),   ### Middle fidelity layers
                    self.geom.random_points(self.num_hi, "sobol"),   ### HIGH fidelity layers

                )
            )
        y_lo_train = self.func_lo(self.X_train)

        y_mi_train = self.func_mi(self.X_train)   ### Middle fidelity layers
        y_hi_train = self.func_hi(self.X_train)   ### HIGH fidelity layers

        self.y_train = [y_lo_train, y_mi_train, y_hi_train]
        return self.X_train, self.y_train

    @run_if_any_none("X_test", "y_test")
    def test(self):
        self.X_test = self.geom.uniform_points(self.num_test, True)
        y_lo_test = self.func_lo(self.X_test)

        y_mi_test = self.func_mi(self.X_test)    ### Middle fidelity layers
        y_hi_test = self.func_hi(self.X_test)    ### HIGH fidelity layers

        self.y_test = [y_lo_test, y_mi_test, y_hi_test]
        return self.X_test, self.y_test




#---------------------------------------------------------------------------------------------------------



class MfData_LMH(Data):
    """Multifidelity function approximation from data set.
        Data with one low fidelity, one middle fidelity and one high fidelity datasets, 
        which will be fed to the LMH Multi-fidelity NN. 

    Args:
        col_x: List of integers.
        col_y: List of integers.
    """

    def __init__(
        self,
        X_lo_train=None,

        X_mi_train=None,      ##### Add Middle fidelity datasets
        X_hi_train=None,      ##### Add HIGH fidelity datasets

        y_lo_train=None,

        y_mi_train=None,      #### Add Middle fidelity datasets
        y_hi_train=None,      #### Add HIGH fidelity datasets

        X_hi_test=None,
        y_hi_test=None,

        fname_lo_train=None,

        fname_mi_train=None,    ##### Add Middle fidelity datasets
        fname_hi_train=None,    ##### Add HIGH fidelity datasets

        # fname_hi_train=None,
        fname_hi_test=None,
        col_x=None,
        col_y=None,
    ):
        if X_lo_train is not None: 
            self.X_lo_train = X_lo_train
            self.X_mi_train = X_mi_train      ##### Add Middle fidelity datasets
            self.X_hi_train = X_hi_train      ##### Add HIGH fidelity datasets

            self.y_lo_train = y_lo_train
            self.y_mi_train = y_mi_train          ##### Add two HIGH fidelity datasets
            self.y_hi_train = y_hi_train          ##### Add two HIGH fidelity datasets

            self.X_hi_test = X_hi_test
            self.y_hi_test = y_hi_test
        elif fname_lo_train is not None:
            data = np.loadtxt(fname_lo_train)     
            self.X_lo_train = data[:, col_x]      
            self.y_lo_train = data[:, col_y]      
          
            data = np.loadtxt(fname_mi_train)
            self.X_mi_train = data[:, col_x]
            self.y_mi_train = data[:, col_y]

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
        n1 = tf.cond(model.net.training, lambda: len(self.X_lo_train), lambda: 0)
        n2 = tf.cond(model.net.training, lambda: len(self.X_mi_train), lambda: 0)

        print("########################################")
        print("n1 is: ", n1)
        print("n2 is: ", n2)

        # loss_lo = loss(targets[0][:n], outputs[0][:n])
        loss_lo = loss(targets[0][: n1], outputs[0][: n1])            ##### Add two fidelity datasets

        loss_mi = loss(targets[1][n1: n1 + n2], outputs[1][n1: n1 + n2])      ##### Add two fidelity datasets
        loss_hi = loss(targets[2][n1 + n2:], outputs[2][n1 + n2:])

        return [loss_lo, loss_mi, loss_hi]

    @run_if_any_none("X_train", "y_train")
    def train_next_batch(self, batch_size=None):
        self.X_train = np.vstack((self.X_lo_train, self.X_mi_train, self.X_hi_train))

        self.y_lo_train, self.y_mi_train, self.y_hi_train = (
            np.vstack((self.y_lo_train, np.zeros_like(self.y_mi_train), np.zeros_like(self.y_hi_train))),   ##### Add
            np.vstack((np.zeros_like(self.y_lo_train), self.y_mi_train, np.zeros_like(self.y_hi_train))),   ##### Add
            np.vstack((np.zeros_like(self.y_lo_train), np.zeros_like(self.y_mi_train), self.y_hi_train)),   ##### Add
        )
        self.y_train = [self.y_lo_train, self.y_mi_train, self.y_hi_train]
        return self.X_train, self.y_train

    def test(self):
        return self.X_hi_test, [self.y_hi_test, self.y_hi_test, self.y_hi_test]

    def _standardize(self):
        self.scaler_x = preprocessing.StandardScaler(with_mean=True, with_std=True)
        self.X_lo_train = self.scaler_x.fit_transform(self.X_lo_train)

        self.X_mi_train = self.scaler_x.fit_transform(self.X_mi_train)     ##### Add two fidelity datasets
        self.X_hi_train = self.scaler_x.fit_transform(self.X_hi_train)     ##### Add two fidelity datasets

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
