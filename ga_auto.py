import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt
import pickle
from func import MLPregression
import geatpy as ea
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from autogluon.features.generators import AutoMLPipelineFeatureGenerator as generator
from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np
import time


# input x column infomation, assign to Vars to avoid errors
column = ['Si_thk', 't_SiO2', 't_polySi_rear_P', 'front_junc', 'rear_junc',
       'resist_rear', 'Nd_top', 'Nd_rear', 'Nt_polySi_top', 'Nt_polySi_rear',
       'Dit Si-SiOx', 'Dit SiOx-Poly', 'Dit top']

# This is a single-objective optimization
# modify the optimization target according to requirements

target='Eff'
# target='FF'
# target='Jsc'
# target='Voc'

output_column = column+[target] 

class MyProblem(ea.Problem):  # Inherit from Problem parent class
    def __init__(self, M=1):
        name = 'MyProblem'  # Initialize name (function name, can be set arbitrarily)
        Dim = 13 # Initialize Dim (decision variable dimensions)
        maxormins = [-1] * M  # Initialize maxormins (objective minimization/maximization marker list, 1: minimize the objective; -1: maximize the objective)
        varTypes = [0] * Dim  # Initialize varTypes (types of decision variables, 0: real number; 1: integer)
        #Si_thk,t_SiO2,t_polySi_rear_P,front_junc,rear_junc,resist_rear,Nd_top,Nd_rear,Nt_polySi_top,Nt_polySi_rear,Dit Si-SiOx,Dit SiOx-Poly,Dit top
        lb = [100, 0.0008, 0.02, 0.3, 0.3, 0.01, 1e18, 1e18, 1e14, 1e14, 1e10, 1e10, 1e10]  # Lower bounds of decision variables
        ub = [200, 0.002,  0.15,   2,   2,   2, 1e21, 1e23, 1e19, 1e20, 1e24, 1e24, 1e24]  # Upper bounds of decision variables

        lbin = [1] * Dim  # Lower boundary of decision variables (0 means not including the lower boundary, 1 means including)
        ubin = [1] * Dim  # Upper boundary of decision variables (0 means not including the upper boundary, 1 means including)
        # Call the parent class constructor method to complete instantiation
        ea.Problem.__init__(self,name,M,maxormins,Dim,varTypes,lb,ub,lbin,ubin)

    def evalVars(self, Vars):  # Objective function
        Vars = pd.DataFrame(Vars)# First convert to dataframe to eliminate scaler warnings
        Vars.columns = column
        Vars = np.array(Vars)
        x = pd.DataFrame(Vars)
        x.columns = column

        p = TabularPredictor.load('Models//final_v3//'+target)
        f = np.array(p.predict(x)).reshape(-1, 1)

        return f
    
if __name__ == '__main__':
    # Instantiate problem object
    problem = MyProblem()
    Encoding='RI'
    # Build algorithm

    algorithm = ea.soea_SEGA_templet(problem,ea.Population(Encoding='RI', NIND=100),
                MAXGEN=300,  # Maximum evolution generations
                logTras=10,  # Indicates how many generations to record log information, 0 means no recording
                #trappedValue=1e-6,  # Threshold for determining stagnation in single-objective optimization
                maxTrappedCount=10# Maximum limit of evolution stagnation counter
                )  
    algorithm.mutOper.F = 0.3  # Parameter F in differential evolution
    algorithm.mutOper.Pm = 0.4  # Mutation probability of mutation operator
    algorithm.recOper.XOVR = 0.8  # Crossover probability of crossover operator

    
    # Solve and output results
    import os
    if not os.path.exists('./output/ga_result'):
        # if path not exist, create the folder
        os.makedirs('./output/ga_result')
    res = ea.optimize(algorithm,
                      #prophet=prophetVars,
                      verbose=True,
                      drawing=3,
                      outputMsg=True,
                      drawLog=True,
                      dirName='output/ga_result/result of '+target+'_ga_auto',
                      title=target,
                      saveFlag=True)
#============================Output Optimization Results===========================

x,y=res['Vars'],res['ObjV']
x_val = x.T
x,y=pd.DataFrame(x),pd.DataFrame(y)
ga_result = pd.concat([x,y],axis=1)
ga_result.columns = output_column
pd.DataFrame(x_val).to_csv(f'output/ga_result/result of {target}_ga_auto/val_para.csv',index=None,header=None)# Generate parameter combinations for tcad validation
ga_result.to_csv(f'output/ga_result/result of {target}_ga_auto/ga_auto_result.csv',index=None)# Save final results

#===================================================================