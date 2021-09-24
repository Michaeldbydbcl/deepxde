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
    
    for x in np.arange(0, Data_Number+1, 1):
        f.write("{:.8f}".format(x/Data_Number) + "  " + "{:.8f}".format(Function(x/Data_Number)))
        f.write('\n')
    f.close()


# Define functions with two levels fidelities, two low and one high 
def func_lo_one(x):
    return 3 * x
                                          
def func_lo_two(x):
    return np.sin(2 * np.pi * x)          
                                          
def func_hi(x):                           
    return 3 * x * np.sin(2 * np.pi * x)  


Write_Data("dataset\mf_lo_train.dat", 100, func_lo_one, "w")
# Write_Data("dataset\mf_lo_train.dat", 50, func_lo_two, "a")
Write_Data("dataset\mf_hi_train.dat", 4, func_hi, "w")
Write_Data("dataset\mf_hi_test.dat", 1000, func_hi, "w")