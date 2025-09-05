import numpy as np
import pandas as pd

def gini_impurity(feature,target):
    df = pd.DataFrame({'feature':feature,'target':target})
    df.sort_values(by='feature',inplace = True)
    table = pd.crosstab(df['target'],df['feature'])
    probability = table.loc[1]/table.sum()
    probability.sort_values(inplace = True)

gini_impurity([1,4,1,4,1,4,1,4,1,1,1,1],[1,0,0,0,1,0,1,0,1,0,1,0])