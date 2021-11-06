from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../../')

import deepxdeNEW as dde
#--------------------------------------------------------------------------------
# Import necessary functions from "Code" folder to process the data
import sys
import math

sys.path.append('../')
import Code as Code

# Import data process function from Code folder, Data_Process.py
from Code.Data_Process import Get_Data, Write_Data

# Import the L2 error function from Code folder, Error_Func.
from Code.Error_Func import L2_RE, RMSE

#--------------------------------------------------------------------------------


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
    activation = "selu"        # (connected lines) 
    # activation = "silu"

    initializer = "Glorot uniform"
    regularization = ["l2", 0.0001]
    net = dde.maps.MfNN_L2H(
        [1] + [10] * 4 + [1],
        [1] + [10] * 4 + [1],
        [40] * 2 + [1],
        activation,
        initializer,
        regularization=regularization,
    )

    model = dde.Model(data, net)
    model.compile("adam", lr=0.0001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=20000)

    # mape.append(dde.utils.apply(mfnn, (data,))[0])      ##### Changed in New version
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    
    # Write data to the file
    L2Error = Get_Data("loss.dat", -1, None)
    f.write("{:.8f}".format(i+1) + "  " + "{:.8f}".format(L2Error[-1][0]))
    # f.write(L2Error[-1])
    f.write('\n')
    print(L2Error)
    print(L2Error[-1])



if __name__ == "__main__":
    #--------------------------------------------------------------------------------
    # Generate several datasets that have different sizes
     
    def func_lo_one(x):
        return x**2 
        # return np.sin(8*np.pi*x + np.pi/10)          
                                            
    def func_lo_two(x):
        # return np.sin(8*np.pi*x + np.pi/10)          
        return np.sin(8*np.pi*x + np.pi/10)**2
                                            
    def func_hi(x):                           
        return x**2 + np.sin(8*np.pi*x + np.pi/10)**2

    Write_Data("dataset\mf_lo_one_train.dat", 200, func_lo_one, "w")
    Write_Data("dataset\mf_lo_two_train.dat", 200, func_lo_two, "w")
    Write_Data("dataset\mf_hi_train.dat", 15, func_hi, "w")
    Write_Data("dataset\mf_hi_test.dat", 1000, func_hi, "w")
    #--------------------------------------------------------------------------------

    f = open("mf_L2Error_Ex2.dat", "w").close()     # Firstly, clear the content of the file
    # main() in dde, postprocessing file -> line 46, don't plot the results
    for i in range(4, 27, 2):
        Write_Data("dataset\mf_hi_train.dat", i, func_hi, "w")
        f = open("mf_L2Error_Ex2.dat", "a")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("Data size is :", i+1)

        main()
    
    x_Error = Get_Data("mf_L2Error_Ex2.dat", 0, 1)
    y_Error = Get_Data("mf_L2Error_Ex2.dat", 1, 2)

    plt.plot(x_Error, y_Error, "b")

    # plt.show()

