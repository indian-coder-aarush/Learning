import numpy as np
import pandas as pd

def gini_impurity(feature,target):
    df = pd.DataFrame({'feature':feature,'target':target})
    df.sort_values(by='feature',inplace = True)
    table = pd.crosstab(df['target'],df['feature'])
    probability_vectors = table/table.sum()
    calculate_probability_class = probability_vectors.idxmax()
    probability = ([table.loc[x, y] for x, y in zip(calculate_probability_class.tolist(),
                                                    calculate_probability_class.index.tolist())]/table.sum())
    probability.sort_values(inplace = True)
    gini_impurities = {}
    for i in range(len(probability)-1):
        right = probability.iloc[i+1:]
        left = probability.iloc[:i+1]
        left_total = table.loc[:,left.index.tolist()].sum().sum()
        right_total = table.loc[:,right.index.tolist()].sum().sum()
        left_class_frequencies = table.loc[:,left.index.tolist()].sum()
        right_class_frequencies = table.loc[:,right.index.tolist()].sum()
        left_impurity = 1 - ((left_class_frequencies/left_total)**2).sum()
        right_impurity = 1 - ((right_class_frequencies/right_total)**2).sum()
        total_impurity = left_impurity*(left_total/(right_total+left_total)) + right_impurity*(right_total/
                                                                                               (right_total+left_total))
        gini_impurities[i+1] = total_impurity
    least_impurity_key = min(gini_impurities)
    split = [probability.iloc[least_impurity_key+1:].index.tolist(),
             probability.iloc[:least_impurity_key+1].index.tolist()]
    return gini_impurities[least_impurity_key] , split

class Node:

    def __init__(self,feature,target,max_depth):
        self.feature = feature
        self.target = target
        self.max_depth = max_depth
        self.left_child = None
        self.right_child = None

