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


# Extract data
x_test = Get_Data("dataset\mf_hi_test.dat", 0, 1)
y_test = Get_Data("dataset\mf_hi_test.dat", 1, 2)

x_low_one_train = Get_Data("dataset\mf_lo_one_train.dat", 0, 1)
y_low_one_train = Get_Data("dataset\mf_lo_one_train.dat", 1, 2)

x_low_two_train = Get_Data("dataset\mf_lo_two_train.dat", 0, 1)
y_low_two_train = Get_Data("dataset\mf_lo_two_train.dat", 1, 2)

x_hi_train = Get_Data("dataset\mf_hi_train.dat", 0, 1)
y_hi_train = Get_Data("dataset\mf_hi_train.dat", 1, 2)

y_predict = Get_Data("test.dat", 4, 7)     # x using high test data


# Plot the train data with predicted data
plt.plot(x_low_one_train, y_low_one_train, label='Low fidelity 1 true')
plt.plot(x_test, y_predict[:,0], label='Low fidelity predict', linestyle='dashed')

plt.plot(x_low_two_train, y_low_two_train, label='Low fidelity 2 true')
plt.plot(x_test, y_predict[:,1], label='Middle fidelity predict', linestyle='dashed')

plt.scatter(x_hi_train, y_hi_train, label='High fidelity train')
plt.plot(x_test, y_predict[:,2], label='High fidelity predict', linestyle='dashed', color='green')
plt.plot(x_test, y_test, label='High fidelity true', color='red')

plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()

# PLot the comparison between mf and GP
DataSize = Get_Data("mf_L2Error_Ex2.dat", 0, 1)

mf_Error = Get_Data("mf_L2Error_Ex2.dat", 1, 2)
GP_Error_Low_One = Get_Data("GP_L2Error_Ex2_High_Change_Low_One_Fix.dat", 1, 2)
GP_Error_Low_Two = Get_Data("GP_L2Error_Ex2_High_Change_Low_Two_Fix.dat", 1, 2)

plt.plot(DataSize, mf_Error, label='L2H L2 error')
plt.plot(DataSize, GP_Error_Low_One, label='GP L2 error, low fidelity one')
plt.plot(DataSize, GP_Error_Low_Two, label='GP L2 error, low fidelity two')

plt.legend()

plt.xlabel("High fidelity dataset size")
plt.ylabel("L2 relativeerror")

plt.title("Comparison between L2H and GP, Ex1")
plt.yscale('log')

plt.show()