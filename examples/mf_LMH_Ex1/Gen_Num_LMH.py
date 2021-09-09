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
def func_lo(x):
    return 3 * x
                                          
def func_mi(x):
    return np.sin(2 * np.pi * x)          
                                          
def func_hi(x):                           
    return 3 * x * np.sin(2 * np.pi * x)  


Write_Data("dataset\mf_lo_train.dat", 200, func_lo)
Write_Data("dataset\mf_mi_train.dat", 200, func_mi)
Write_Data("dataset\mf_hi_train.dat", 10, func_hi)
Write_Data("dataset\mf_hi_test.dat", 1000, func_hi)