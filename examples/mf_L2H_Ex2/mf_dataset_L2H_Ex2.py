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
    regularization = ["l2", 0.00005]   # change to small value?
    net = dde.maps.MfNN_L2H(
        [1] + [900] * 4 + [1],
        [1] + [700] * 4 + [1],
        [60] * 2 + [1],
        activation,
        initializer,
        regularization=regularization,
    )
    # Note, 800, 800, 50 works fine, error to 8%.
    # Note, 800, 800, 55 down to 8%, but bounces back
    # Note, 900, 800, 50 goes to 3.60%.
    # Note, 900, 700, 50 goes to 2.12%. (Reg = 0.0001)
    # Note, 900, 700, 60, reg = 0.00005, lr=0.0001 can go down to 1.45%,, high train 20 points, more epoches does not help
    # Note, 900, 700, 60, reg = 0.00007, lr=0.0001 can go down to 7.00%, more epoches does not help
    # Note, 900, 700, 60, reg = 0.00004, lr=0.0001 to 10.00%, , high trian points around 10, more epoches does not help


    model = dde.Model(data, net)
    model.compile("adam", lr=0.0001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=500000)

    # mape.append(dde.utils.apply(mfnn, (data,))[0])      ##### Changed in New version

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
