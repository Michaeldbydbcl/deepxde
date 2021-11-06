from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import GPy
import matplotlib.pyplot as plt
import numpy as np
import emukit.multi_fidelity
import emukit.test_functions
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel

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


class LinearMFGP(object):
    def __init__(self, noise=None, n_optimization_restarts=10):
        self.noise = noise
        self.n_optimization_restarts = n_optimization_restarts
        self.model = None

    def train(self, x_l, y_l, x_h, y_h):
        # Construct a linear multi-fidelity model
        X_train, Y_train = convert_xy_lists_to_arrays([x_l, x_h], [y_l, y_h])
        kernels = [GPy.kern.RBF(x_l.shape[1]), GPy.kern.RBF(x_h.shape[1])]
        kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
        gpy_model = GPyLinearMultiFidelityModel(
            X_train, Y_train, kernel, n_fidelities=2
        )
        if self.noise is not None:
            gpy_model.mixed_noise.Gaussian_noise.fix(self.noise)
            gpy_model.mixed_noise.Gaussian_noise_1.fix(self.noise)

        # Wrap the model using the given 'GPyMultiOutputWrapper'
        self.model = GPyMultiOutputWrapper(
            gpy_model, 2, n_optimization_restarts=self.n_optimization_restarts
        )
        # Fit the model
        self.model.optimize()

    def predict(self, x):
        # Convert x_plot to its ndarray representation
        X = convert_x_list_to_array([x, x])
        X_l = X[: len(x)]
        X_h = X[len(x) :]

        # Compute mean predictions and associated variance
        lf_mean, lf_var = self.model.predict(X_l)
        lf_std = np.sqrt(lf_var)
        hf_mean, hf_var = self.model.predict(X_h)
        hf_std = np.sqrt(hf_var)
        return lf_mean, lf_std, hf_mean, hf_std


def main():

##### From previous example of Gaussian process
    # high_fidelity = emukit.test_functions.forrester.forrester
    # low_fidelity = emukit.test_functions.forrester.forrester_low
    # x_plot = np.linspace(0, 1, 200)[:, None]
    # y_plot_l = low_fidelity(x_plot)
    # y_plot_h = high_fidelity(x_plot)    
    # x_plot_l = Get_Data("dataset\mf_lo_one_train.dat", 0, 1)
    # y_plot_l = Get_Data("dataset\mf_lo_one_train.dat", 1, 2)
    x_plot_l = Get_Data("dataset\mf_lo_train.dat", 0, 1)
    y_plot_l = Get_Data("dataset\mf_lo_train.dat", 1, 2)
  
    x_plot_h = Get_Data("dataset\mf_hi_train.dat", 0, 1)
    y_plot_h = Get_Data("dataset\mf_hi_train.dat", 1, 2)

    x_plot_h_True = Get_Data("dataset\mf_hi_test.dat", 0, 1)
    y_plot_h_True = Get_Data("dataset\mf_hi_test.dat", 1, 2)

    # x_train_h = np.atleast_2d(np.random.permutation(x_train_l)[:8])
    # y_train_l = low_fidelity(x_train_l)
    # y_train_h = high_fidelity(x_train_h)
    x_train_l = Get_Data("dataset\mf_lo_train.dat", 0, 1)
    y_train_l = Get_Data("dataset\mf_lo_train.dat", 1, 2)
    x_train_h = Get_Data("dataset\mf_hi_train.dat", 0, 1)
    y_train_h = Get_Data("dataset\mf_hi_train.dat", 1, 2)

    model = LinearMFGP(noise=0, n_optimization_restarts=10)
    model.train(x_train_l, y_train_l, x_train_h, y_train_h)
    lf_mean, lf_std, hf_mean, hf_std = model.predict(x_plot_l)
    lf_mean_cal, lf_std_cal, hf_mean_cal, hf_std_cal = model.predict(x_plot_h)
    lf_mean_cal, lf_std_cal, hf_mean_cal, hf_std_cal = model.predict(x_plot_h_True)


#---------------------------------------------------------------------
##### Print out the predicted values and the ture values

    print("The predicted low values are: ", lf_mean_cal)
    print("The true low values are: ", y_train_l)
    print("#-------------------------------------------------------")
    print("The predicted high values are: ", hf_mean_cal)
    print("The true high values are: ", y_train_h)
    print("#-------------------------------------------------------")

    print("L2 relative error for high train is: ", L2_RE(hf_mean, y_train_h))
    L2Error = L2_RE(hf_mean_cal, y_plot_h_True)
    f.write("{:.8f}".format(i+1) + "  " + "{:.8f}".format(L2Error))
    f.write('\n')

    print("Root mean squared error for high train is: ", RMSE(hf_mean, y_train_h))

    print("L2 relative error for low train is:  ", L2_RE(lf_mean, y_train_l))

#---------------------------------------------------------------------

    # plt.figure(figsize=(12, 8))
    # plt.plot(x_plot_l, y_plot_l, "b")
    # plt.plot(x_plot_h, y_plot_h, "r")
    # plt.scatter(x_train_l, y_train_l, color="b", s=40)
    # plt.scatter(x_train_h, y_train_h, color="r", s=40)
    # plt.ylabel("f (x)")
    # plt.xlabel("x")
    # plt.legend(["Low fidelity", "High fidelity"])
    # plt.title("High and low fidelity Forrester functions")
# 
    # Plot the posterior mean and variance
    # plt.figure(figsize=(12, 8))
    # plt.fill_between(
        # x_plot_l.flatten(),
        # (lf_mean - 1.96 * lf_std).flatten(),
        # (lf_mean + 1.96 * lf_std).flatten(),
        # facecolor="g",
        # alpha=0.3,
    # )
    # plt.fill_between(
        # x_plot_l.flatten(),
        # (hf_mean - 1.96 * hf_std).flatten(),
        # (hf_mean + 1.96 * hf_std).flatten(),
        # facecolor="y",
        # alpha=0.3,
    # )
# 
    # plt.plot(x_plot_l, y_plot_l, "b")
    # plt.plot(x_plot_h, y_plot_h, "r")
    # plt.plot(x_plot_l, lf_mean, "--", color="g")
    # plt.plot(x_plot_h_True, hf_mean_cal, "--", color="y")
    # plt.scatter(x_train_l, y_train_l, color="b", s=40)
    # plt.scatter(x_train_h, y_train_h, color="r", s=40)
    # plt.ylabel("f (x)")
    # plt.xlabel("x")
    # plt.legend(
    #     [
    #         "Low Fidelity",
    #         "High Fidelity",
    #         "Predicted Low Fidelity",
    #         "Predicted High Fidelity",
    #     ]
    # )
    # plt.title(
    #     "Linear multi-fidelity model fit to low and high fidelity function"
    # )
    # plt.show()


if __name__ == "__main__":
    #--------------------------------------------------------------------------------
    # Generate several datasets that have different sizes
    
    def func_lo_one(x):
        return 3 * x
                                            
    def func_lo_two(x):
        return np.sin(2 * np.pi * x)          
                                            
    def func_hi(x):                           
        return 3 * x * np.sin(2 * np.pi * x)  
    
    Write_Data("dataset\mf_lo_train.dat", 100, func_lo_one, "w")

    Write_Data("dataset\mf_hi_train.dat", 15, func_hi, "w")
    Write_Data("dataset\mf_hi_test.dat", 1000, func_hi, "w")


    #--------------------------------------------------------------------------------
    # Below fixes the low fidelity data size with low one function and vary the high fidelity
    f = open("GP_L2Error_Ex1_High_Change_Low_One_Fix.dat", "w").close()     # Firstly, clear the content of the file

    for i in range(4, 25, 1):
        Write_Data("dataset\mf_hi_train.dat", i, func_hi, "w")
        f = open("GP_L2Error_Ex1_High_Change_Low_One_Fix.dat", "a")

        main()
    
    x_Error = Get_Data("GP_L2Error_Ex1_High_Change_Low_One_Fix.dat", 0, 1)
    y_Error = Get_Data("GP_L2Error_Ex1_High_Change_Low_One_Fix.dat", 1, 2)

    plt.plot(x_Error, y_Error, "b")

    plt.show()


    #--------------------------------------------------------------------------------
    Write_Data("dataset\mf_lo_train.dat", 100, func_lo_two, "w")

    # Below fixes the low fidelity data size with low two function and vary the high fidelity
    f = open("GP_L2Error_Ex1_High_Change_Low_Two_Fix.dat", "w").close()     # Firstly, clear the content of the file

    for i in range(4, 25, 1):
        Write_Data("dataset\mf_hi_train.dat", i, func_hi, "w")
        f = open("GP_L2Error_Ex1_High_Change_Low_Two_Fix.dat", "a")

        main()
    
    x_Error = Get_Data("GP_L2Error_Ex1_High_Change_Low_Two_Fix.dat", 0, 1)
    y_Error = Get_Data("GP_L2Error_Ex1_High_Change_Low_Two_Fix.dat", 1, 2)

    plt.plot(x_Error, y_Error, "b")

    plt.show()