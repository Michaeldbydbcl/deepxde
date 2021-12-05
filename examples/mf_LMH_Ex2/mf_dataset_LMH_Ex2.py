from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../../')

import deepxdeNEW as dde


def main():
    fname_lo_train = "dataset/mf_lo_train.dat"

    fname_mi_train = "dataset/mf_mi_train.dat"     ##### Add Middle fidelity datasets
    fname_hi_train = "dataset/mf_hi_train.dat"     ##### Add HIGH fidelity datasets

    fname_hi_test = "dataset/mf_hi_test.dat"

    ##### Plot the data loaded from the files
    # data_temp = np.loadtxt(fname_lo_one_train)
    # x_lo_one_train_temp = data_temp[:,0]
    # y_lo_one_train_temp = data_temp[:,1]
# 
    # plt.scatter(x_lo_one_train_temp, y_lo_one_train_temp)
    # plt.show()
# 
    # data_temp = np.loadtxt(fname_lo_two_train)
    # x_lo_two_train_temp = data_temp[:,0]
    # y_lo_two_train_temp = data_temp[:,1]
# 
    # plt.scatter(x_lo_two_train_temp, y_lo_two_train_temp)
    # plt.show()
# 
    # data_temp = np.loadtxt(fname_hi_train)
    # x_hi_train_temp = data_temp[:,0]
    # y_hi_train_temp = data_temp[:,1]
# 
    # plt.scatter(x_hi_train_temp, y_hi_train_temp)
    # plt.show()
# 
    # data_temp = np.loadtxt(fname_hi_test)
    # x_hi_test_temp = data_temp[:,0]
    # y_hi_test_temp = data_temp[:,1]
    # 
    # plt.scatter(x_hi_test_temp, y_hi_test_temp)
# 
    # plt.show() 
    #####-------------------- finish plot
   


    data = dde.data.MfData_LMH(
        fname_lo_train=fname_lo_train,
        fname_mi_train=fname_mi_train,
        fname_hi_train=fname_hi_train,
        fname_hi_test=fname_hi_test,
        col_x=(0,),
        col_y=(1,),
    )

    activation = "tanh"
    initializer = "Glorot uniform"
    regularization = ["l2", 0.0001]
    net = dde.maps.MfNN_LMH(
        [1] + [400] * 4 + [1],
        [1] + [200] * 4 + [1],
        [100] * 2 + [1],
        activation,
        initializer,
        regularization=regularization,
    )
    # 600, 400, 50, reg=0.00001, lr=0.005, training set works,  test set not very good, overfitting?
    # 100, 80, 500, reg=0.00001 with
        # loss_weights=(1,0,0,1), epochs=60000
        # loss_weights=(0,1,0,1), epochs=60000
        # loss_weights=(0,0,1,1), epochs=100000
        # can go down to 2% error.
    

    model = dde.Model(data, net)
    # model.compile("adam", lr=0.0001, loss="MAPE", metrics=["MAPE", "APE SD"])

    model.compile("adam", lr=0.0001, loss_weights=(1, 1, 1, 1), metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=120000)

    # model.compile("adam", lr=0.0001, loss_weights=(1, 0, 0, 1), loss="MAPE", metrics=["MAPE", "APE SD"])
    # losshistory, train_state = model.train(epochs=60000)
# 
    # model.compile("adam", lr=0.0001, loss_weights=(0, 1, 0, 1), loss="MAPE", metrics=["MAPE", "APE SD"])
    # losshistory, train_state = model.train(epochs=60000)
# 
    # model.compile("adam", lr=0.0001, loss_weights=(0, 0, 1, 1), loss="MAPE", metrics=["MAPE", "APE SD"])
    # losshistory, train_state = model.train(epochs=10000)

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
