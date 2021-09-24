'''
    This file generate two set of low fidelity data and one set of high fidelity data.
    Using a high fidelity function as test function.
'''

import numpy as np
import pandas
import math

###-----------------------------------------------
### Define function that write two columns of data
###-----------------------------------------------

def Write_Data(File_Name, Data_Number, Function, Write_Type):
    f = open(File_Name, Write_Type)
    # f.write("# x, y" + '\n')
            
    for x in np.arange(0, Data_Number+1, 1):
        f.write("{:.8f}".format(x/Data_Number) + "  " + "{:.8f}".format(Function(x/Data_Number)))
        f.write('\n')
    f.close()


###-----------------------------------------------
### Define L^2 relative error formula
###-----------------------------------------------

def L2_RE(Data_Pred, Data_True):
    sum_error_squared = 0
    sum_true_squared = 0

    for i in range(len(Data_True)):
        sum_error_squared += (Data_True[i] - Data_Pred[i])**2/len(Data_True)
        sum_true_squared += Data_True[i]**2

    print(sum_error_squared)
    print(sum_true_squared)
    L2RE = math.sqrt(sum_error_squared/sum_true_squared) 

    return L2RE

print("test case error is: ", L2_RE([1, 2, 3], [1.1, 2.1, 3.1]))

###-----------------------------------------------------------------
# Define functions with two levels fidelities, two low and one high 
###-----------------------------------------------------------------

def func_lo_one(x):
    return 3 * x
                                          
def func_lo_two(x):
    return np.sin(2 * np.pi * x)          
                                          
def func_hi(x):                           
    return 3 * x * np.sin(2 * np.pi * x)  


Write_Data("dataset\mf_lo_one_train.dat", 100, func_lo_one, "w")
Write_Data("dataset\mf_lo_two_train.dat", 10, func_lo_two, "w")
Write_Data("dataset\mf_hi_train.dat", 6, func_hi, "w")
Write_Data("dataset\mf_hi_test.dat", 1000, func_hi, "w")