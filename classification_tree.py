import numpy as np

def gini_impurity(labels):
    total = labels.sum()
    gini_impurity = 1
    for i in range(labels):
        gini_impurity -= (labels[i] / total)**2
    return gini_impurity

