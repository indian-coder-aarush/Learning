import numpy as np
import pandas as pd

def gini_impurity(feature,target):
    df = pd.DataFrame({'feature':feature,'target':target})
    df.sort_values(by='feature',inplace = True)
    table = pd.crosstab(df['target'],df['feature'])
    probability = table.loc[1]/table.sum()
    probability.sort_values(inplace = True)
    gini_impurities = {}
    for i in range(len(probability)-1):
        right = probability.iloc[i+1:]
        left = probability.iloc[:i+1]
        left_total = table.loc[:,left.index.tolist()].sum().sum()
        right_total = table.loc[:,right.index.tolist()].sum().sum()
        left_1s = table.loc[1,left.index.tolist()].sum().sum()
        right_1s = table.loc[1,right.index.tolist()].sum().sum()
        left_impurity = 1 - (left_1s/left_total)**2 - (1 - left_1s/left_total)**2
        right_impurity = 1 - (right_1s / right_total) ** 2 - (1 - right_1s / right_total) ** 2
        total_impurity = left_impurity*(left_total/(right_total+left_total)) + right_impurity*(right_total/
                                                                                               (right_total+left_total))
        gini_impurities[i+1] = total_impurity
    least_impurity_key = min(gini_impurities)
    split = [probability.iloc[least_impurity_key+1:].index.tolist(),
             probability.iloc[:least_impurity_key+1].index.tolist()]
    return gini_impurities[least_impurity_key] , split


print(gini_impurity([1,4,1,4,1,4,1,4,1,1,1,1,2,2,2,2,2,2,2,2],[1,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1]))