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

x_lo_train = Get_Data("dataset\mf_lo_train.dat", 0, 1)
y_lo_train = Get_Data("dataset\mf_lo_train.dat", 1, 2)

x_hi_train = Get_Data("dataset\mf_hi_train.dat", 0, 1)
y_hi_train = Get_Data("dataset\mf_hi_train.dat", 1, 2)

y_predict = Get_Data("test.dat", 3, 5)     # x using high test data


# Plot the train data with predicted data
plt.plot(x_lo_train, y_lo_train, label='Low fidelity true')
plt.plot(x_test, y_predict[:,0], label='Low fidelity predict', linestyle='dashed')

plt.scatter(x_hi_train, y_hi_train, label='High fidelity train')
plt.plot(x_test, y_predict[:,1], label='High fidelity predict', linestyle='dashed')
plt.plot(x_test, y_test, label='High fidelity true')

plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()