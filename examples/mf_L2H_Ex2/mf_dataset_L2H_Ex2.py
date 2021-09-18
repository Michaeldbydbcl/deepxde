from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../../')

import deepxdeNEW as dde


def main():
    # fname_lo_train = "dataset/mf_lo_train.dat"
    fname_lo_one_train = "dataset/mf_lo_one_train.dat"
    fname_lo_two_train = "dataset/mf_lo_two_train.dat"

    fname_hi_train = "dataset/mf_hi_train.dat"
    fname_hi_test = "dataset/mf_hi_test.dat"

    # ##### Plot the data loaded from the files
    # data_temp = np.loadtxt(fname_lo_one_train)
    # x_lo_one_train_temp = data_temp[:,0]
    # y_lo_one_train_temp = data_temp[:,1]

    # plt.scatter(x_lo_one_train_temp, y_lo_one_train_temp)
    # plt.show()

    # data_temp = np.loadtxt(fname_lo_two_train)
    # x_lo_two_train_temp = data_temp[:,0]
    # y_lo_two_train_temp = data_temp[:,1]

    # plt.scatter(x_lo_two_train_temp, y_lo_two_train_temp)
    # plt.show()


    # data_temp = np.loadtxt(fname_hi_train)
    # x_hi_train_temp = data_temp[:,0]
    # y_hi_train_temp = data_temp[:,1]

    # plt.scatter(x_hi_train_temp, y_hi_train_temp)
    # plt.show()

    # data_temp = np.loadtxt(fname_hi_test)
    # x_hi_test_temp = data_temp[:,0]
    # y_hi_test_temp = data_temp[:,1]
    
    # plt.scatter(x_hi_test_temp, y_hi_test_temp)

    # plt.show()

    mape = []    ### Print out the MAPE of the results
       
    data = dde.data.MfData_L2H(
        fname_lo_one_train=fname_lo_one_train,
        fname_lo_two_train=fname_lo_two_train,
        fname_hi_train=fname_hi_train,
        fname_hi_test=fname_hi_test,
        col_x=(0,),
        col_y=(1,),
    )

#####  Need to adjust parameters. 
    # activation = "tanh" 
    # activation = "sigmoid"     # (Very bad results)
    # activation = "relu"        # (connected lines) 
    # activation = "selu"        # (connected lines) 
    activation = "tanh"

    initializer = "Glorot uniform"
    regularization = ["l2", 0.001]   # change to small value?
    net = dde.maps.MfNN_L2H(
        [1] + [20] * 4 + [1],
        [1] + [20] * 4 + [1],
        [10] * 2 + [1],
        activation,
        initializer,
        regularization=regularization,
    )

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=20000)

    # mape.append(dde.utils.apply(mfnn, (data,))[0])      ##### Changed in New version

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
