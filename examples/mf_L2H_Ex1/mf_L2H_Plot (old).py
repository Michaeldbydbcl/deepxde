import matplotlib.pyplot as plt
import numpy as np

# Define the function that extrac correct x data
def Get_x(FileName):
    x = []

    with open(FileName) as xFile:
        for each_line in xFile:
            line_split = each_line.split()
            if (line_split[0] != "#"):
                x.append(line_split[0])

    x_string = np.array(x)
    x_float = x_string.astype(np.float64)
    return x_float


# Define the function that plot the high test data
def High_Test_Plot(FileName):
    y_hi = []

    with open(FileName) as PlotFile:
        for each_line in PlotFile:
            line_split = each_line.split()

            if (line_split[0] != "#"):
                y_hi.append(line_split[1])
    
    x_float = Get_x("dataset\mf_hi_test.dat")

    y_hi_string = np.array(y_hi)
    y_hi_float = y_hi_string.astype(np.float64)

    plt.plot(x_float, y_hi_float, marker=".")
    plt.show()


# Define the function that plot the test data and predicted data
def Data_Plot(FileName):
    y_test = []
    y_predict = []

    with open(FileName) as PlotFile:
        for each_line in PlotFile:
            line_split = each_line.split()

            if (line_split[0] != "#"):
                y_test.append(line_split[1: 4])
                y_predict.append(line_split[4: 7])

    x_float = Get_x("dataset\mf_hi_test.dat")
   
    y_test_string = np.array(y_test)
    y_test_float = y_test_string.astype(np.float64)

    y_predict_string = np.array(y_predict)
    y_predict_float = y_predict_string.astype(np.float64)
    
    for i in range(3):
        plt.plot(x_float, y_test_float[:,i], marker=".")
        plt.plot(x_float, y_predict_float[:,i], marker="p")

    plt.show()


# Now plot the test data 
High_Test_Plot("dataset\mf_hi_test.dat")
Data_Plot("test.dat")



