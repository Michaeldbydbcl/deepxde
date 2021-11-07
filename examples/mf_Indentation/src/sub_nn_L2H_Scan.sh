#!/bin/bash

#SBATCH --job-name=nnL2H
#SBATCH --time=02:00:00 
#SBATCH --output=output.%j 
#SBATCH --ntasks=4
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=wshi@usf.edu 

/home/w/wshi/anaconda3/bin/python3.8 nn_L2H_Scan.py