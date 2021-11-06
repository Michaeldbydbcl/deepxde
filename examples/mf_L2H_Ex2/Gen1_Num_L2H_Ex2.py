'''
    This file generate two set of low fidelity data and one set of high fidelity data.
    Using a high fidelity function as test function.
'''

import numpy as np
import pandas

###-----------------------------------------------
### Define function that write two columns of data

def Write_Data(File_Name, Data_Number, Function, Write_Type):
    f = open(File_Name, Write_Type)
    f.write("# x, y" + '\n')
    
    for x in np.arange(0, Data_Number+1, 1):     ### !!!!! No adding 1
        f.write("{:.8f}".format(x/Data_Number) + "  " + "{:.8f}".format(Function(x/Data_Number)))
        f.write('\n')
    f.close()


# Define functions with two levels fidelities, two low and one high 
# This function is a phase-shifted oscillation function

def func_lo_one(x):
    return x**2 
                                          
def func_lo_two(x):
    return np.sin(8*np.pi*x)          
                                          
def func_hi(x):                           
    return x**2 + np.sin(8*np.pi*x + np.pi/10)**2


Write_Data("dataset\mf_lo_one_train.dat", 100, func_lo_one, "w")
Write_Data("dataset\mf_lo_two_train.dat", 100, func_lo_two, "w")
Write_Data("dataset\mf_hi_train.dat", 15, func_hi, "w")
Write_Data("dataset\mf_hi_test.dat", 1000, func_hi, "w")