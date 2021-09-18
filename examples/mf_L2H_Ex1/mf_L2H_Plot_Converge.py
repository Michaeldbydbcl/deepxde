import matplotlib.pyplot as plt
import numpy as np

#-----------------------------------------------------
# Define the function that extract corresponding data

def Get_Data(FileName, Data_Column1, Data_Column2):
    Data_Extract = []

    with open(FileName) as Data_File:
        for each_line in Data_File:
            line_split = each_line.split()
            if (line_split[0] != "#"):
                Data_Extract.append(line_split[Data_Column1:Data_Column2])

    Data_Extract_string = np.array(Data_Extract)
    Data_Extract_float = Data_Extract_string.astype(np.float64)
    return Data_Extract_float


#-------------------------------------------------------
# steps = Get_Data("loss.dat", 0, 1) 
# test_metric = Get_Data("loss.dat", -3, None)

steps_L2H_Ex2 = Get_Data("..\mf_L2H_Ex2\loss.dat", 0, 1) 
test_metric_L2H_Ex2 = Get_Data("..\mf_L2H_Ex2\loss.dat", -3, None)

steps_LMH_Ex2 = Get_Data("..\mf_LMH_Ex2\loss.dat", 0, 1) 
test_metric_LMH_Ex2 = Get_Data("..\mf_LMH_Ex2\loss.dat", -3, None)

steps_LH_Ex2 = Get_Data("..\mf_LH_Ex2\loss.dat", 0, 1) 
test_metric_LH_Ex2 = Get_Data("..\mf_LH_Ex2\loss.dat", -3, None)

plt.plot(steps_L2H_Ex2, test_metric_L2H_Ex2[:,2], color="red", label="L2H")
plt.plot(steps_LMH_Ex2, test_metric_LMH_Ex2[:,2], color="blue", label="LMH")
plt.plot(steps_LH_Ex2,  test_metric_LH_Ex2[:,2] , color="green", label="LH")

plt.legend()
plt.show()
