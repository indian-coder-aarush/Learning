import numpy as np
import pandas as pd

def gini_impurity(feature,target):
    classes = []
    sorted_feature = []
    probabilities = []
    for i in range(len(feature)):
        if not(feature[i] in classes):
            classes.append(feature[i])
    for i in range(len(classes)):
        sorted_feature.append([i for i, value in enumerate(feature) if value == feature[i]])
        probabilities[i] = len(sorted_feature[i]) / len(feature)