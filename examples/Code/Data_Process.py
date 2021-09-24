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


#-----------------------------------------------------
# Define the function that extract corresponding data
#-----------------------------------------------------

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
