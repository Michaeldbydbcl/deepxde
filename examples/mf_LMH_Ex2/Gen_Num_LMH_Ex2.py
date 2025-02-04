'''
    This file generate low, middle and high fidelity datasets
    Using a high fidelity function as test function.
'''

import numpy as np
import pandas

###-----------------------------------------------
### Define function that write two columns of data

def Write_Data(File_Name, Data_Number, Function):
    f = open(File_Name, "w")
    f.write("# x, y" + '\n')
    
    for x in np.arange(0, Data_Number+1, 1):
        f.write("{:.8f}".format(x/Data_Number) + "  " + "{:.8f}".format(Function(x/Data_Number)))
        f.write('\n')
    f.close()


# Define functions with two levels fidelities, two low and one high 
# This function is a phase-shifted oscillation function

def func_lo(x):
    return x**2
                                          
def func_mi(x):
    return np.sin(8*np.pi*x + np.pi/10)          
                                          
def func_hi(x):                           
    return x**2 + np.sin(8*np.pi*x + np.pi/10)


Write_Data("dataset\mf_lo_train.dat", 200, func_lo, "w")
Write_Data("dataset\mf_mi_train.dat", 200, func_mi, "w")
Write_Data("dataset\mf_hi_train.dat", 10,  func_hi, "w")
Write_Data("dataset\mf_hi_test.dat", 1000, func_hi, "w")