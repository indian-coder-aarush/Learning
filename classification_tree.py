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
    for i in range(len(sorted_feature)):
        class_probability = 0
        for j in range(len(sorted_feature[i])):
            if feature[sorted_feature[i][j]] == target[sorted_feature[i][j]]:
                class_probability += 1
        probabilities[i] = class_probability / len(sorted_feature[i])
