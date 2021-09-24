'''
   This file contains some error functions 
''' 
import math

###-----------------------------------------------
### Define L^2 relative error formula
###-----------------------------------------------

def L2_RE(Data_Pred, Data_True):
    sum_error_squared = 0
    sum_true_squared = 0

    for i in range(len(Data_True)):
        sum_error_squared += (Data_True[i] - Data_Pred[i])**2 
        sum_true_squared += Data_True[i]**2

    # print(sum_error_squared)
    # print(sum_true_squared)
    L2RE = math.sqrt(sum_error_squared/sum_true_squared) 

    return L2RE


###-----------------------------------------------
### Define mean square error formula
###-----------------------------------------------

def RMSE(Data_Pred, Data_True):
    sum_error_squared = 0
    sum_true_squared = 0

    for i in range(len(Data_True)):
        sum_error_squared += (Data_True[i] - Data_Pred[i])**2/len(Data_True)

    # print(sum_error_squared)
    RMSE = math.sqrt(sum_error_squared) 

    return RMSE
