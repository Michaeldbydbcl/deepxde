from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import sys

sys.path.append("../../../")

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold, ShuffleSplit

import deepxdeNEW as dde
from data import BerkovichData, ExpData, FEMData, ModelData


def svm(data):
    clf = SVR(kernel="rbf")
    clf.fit(data.train_x, data.train_y[:, 0])
    y_pred = clf.predict(data.test_x)[:, None]
    return dde.metrics.get("MAPE")(data.test_y, y_pred)


def mfgp(data):
    from mfgp import LinearMFGP

    model = LinearMFGP(noise=0, n_optimization_restarts=5)
    model.train(data.X_lo_train, data.y_lo_train, data.X_hi_train, data.y_hi_train)
    _, _, y_pred, _ = model.predict(data.X_hi_test)
    return dde.metrics.get("MAPE")(data.y_hi_test, y_pred)


def nn(data):
    layer_size = [data.train_x.shape[1]] + [32] * 2 + [1]
    activation = "selu"
    initializer = "LeCun normal"
    regularization = ["l2", 0.01]

    loss = "MAPE"
    optimizer = "adam"
    if data.train_x.shape[1] == 3:
        lr = 0.0001
    else:
        lr = 0.001
    epochs = 30000

    net = dde.maps.FNN(
        layer_size, activation, initializer, regularization=regularization
    )
    model = dde.Model(data, net)
    model.compile(optimizer, lr=lr, loss=loss, metrics=["MAPE"])
    losshistory, train_state = model.train(epochs=epochs)
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
    return train_state.best_metrics[0]


def mfnn_L2H(data, LowDim):
    x_dim, y_dim = 3, 1
    # activation = "selu"
    activation = "tanh"

    initializer = "LeCun normal"
    regularization = ["l2", 0.0005]     # Change to smaller value as in function cases
    net = dde.maps.MfNN_L2H(
        [x_dim] + [LowDim] * 2 + [y_dim],
        [x_dim] + [128] * 2 + [y_dim],
        [64] * 2 + [y_dim],
        activation,
        initializer,
        regularization=regularization,
        residue=True,
        trainable_low_one_fidelity=True,
        trainable_low_two_fidelity=True,
        trainable_high_fidelity=True,
    )
    # 256, 128, 16, reg=0.00001, lr=0.00001, "tanh", goes to around 40%.
    # 256, 128, 64, reg=0.00005, lr=0.00001, "tanh", goes to around 10%, with best data 8%.
    # 256, 256, 64, reg=0.00005, lr=0.00001, "tanh", goes to around 15%, with best data 11%.
    # 256, 128, 128, reg=0.00005, lr=0.00001, "tanh", goes to around 20%, with best data 15%.
    # 256, 128, 64, reg=0.00005, lr=0.00001, "tanh", goes down to 10%, then goes back to 20%

    model = dde.Model(data, net)
    model.compile("adam", lr=0.00001, loss="MAPE", metrics=["MAPE", "APE SD"])
    losshistory, train_state = model.train(epochs=80000)



    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
    return (
        train_state.best_metrics[1],
        train_state.best_metrics[3],
        train_state.best_y[1],
    )


def validation_mf(yname, train_size, Run_Number, LowDim):
    datalow_one = FEMData(yname, [70])
    datalow_two = BerkovichData(yname)

    # datalow = ModelData(yname, 10000, "forward_n")
    # datahigh = BerkovichData(yname)
    # datahigh = FEMData(yname, [70])
    datahigh = ExpData("../data/B3067.csv", yname)

    kf = ShuffleSplit(
        n_splits=10, test_size=len(datahigh.X) - train_size, random_state=0
    )
    # kf = LeaveOneOut()

    mape = []
    iter = 0
    for train_index, test_index in kf.split(datahigh.X):
        iter += 1
        print("\nCross-validation iteration: {}".format(iter), flush=True)

        data = dde.data.MfData_L2H(
            X_lo_one_train=datalow_one.X,     ##### Add one more low fidelity dataset
            X_lo_two_train=datalow_two.X,     ##### Add one more low fidelity dataset
            X_hi_train=datahigh.X[train_index],
            y_lo_one_train=datalow_one.y,     ##### Add one more low fidelity dataset
            y_lo_two_train=datalow_two.y,     ##### Add one more low fidelity dataset
            y_hi_train=datahigh.y[train_index],
            X_hi_test=datahigh.X[test_index],
            y_hi_test=datahigh.y[test_index],
        )

        mape.append(dde.utils.apply(mfnn_L2H, (data, LowDim))[0])      ##### "apply" function changed in New version



    ### Open a file to store MAPE values
    with open(yname+"_"+Run_Number+".txt", "a") as myfile:
        Parameter_Chosen = ["Low Dim =", str(LowDim), "\n"]
        L = [yname, ",", str(train_size), ",", str(np.mean(mape)), ",", str(np.std(mape)), "\n"]
        myfile.writelines(Parameter_Chosen)        
        myfile.writelines(L)
        
    print(mape)
    print(yname, train_size, np.mean(mape), np.std(mape))

def main():
    # validation_FEM("E*", [50, 60, 70, 80], 70)
    # validation_mf("E*", 9)
    # validation_scaling("E*")
    # validation_exp("E*")
    # validation_exp_cross("E*")
    # validation_exp_cross2("E*", 10)
    # validation_exp_cross3("E*")
    # validation_exp_cross_transfer("E*")
    # return

    # file = open("C:\\Users\\wshi\\Documents\\GitHub\\deep-learning-for-indentation\\src\\MAPE.txt", "w")
    for i in range(1, 12):
        # validation_model("E*", train_size)
        # validation_FEM("sigma_y", [50, 60, 70, 80], train_size)

        # file.write(validation_mf("E*", train_size))
        validation_mf("sigma_y", 1, "L2H_run1", 256)
        # validation_exp_cross2("E*", train_size)

        print("=======================================================")
        print("=======================================================")

    # file.close()


if __name__ == "__main__":
    main()
