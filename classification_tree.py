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

    def __init__(self,feature,target,max_depth,depth):
        self.feature = feature
        self.target = target
        self.max_depth = max_depth
        self.depth = depth
        self.left_child = None
        self.right_child = None
        self.split = None
        self.split_feature_index = None

    def make_children(self):
        if self.left_child is None and self.right_child is None and self.depth > self.max_depth:
            feature = pd.DataFrame(self.feature)
            target = pd.DataFrame(self.target)
            for i in range(len(self.feature)):
                gini_impurities = []
                splits = []
                take_input_tuple = gini_impurity(self.feature[i],self.target)
                gini_impurities.append(take_input_tuple[0])
                splits.append(take_input_tuple[1])
            if min(gini_impurities) < 0:
                return
            split_index = gini_impurities.index(min(gini_impurities))
            split = splits[split_index]
            self.split = split
            self.split_feature_index = split_index
            self.left_child = Node(feature[feature[split_index] in split[0]],target[feature[split_index] in split[0]],
                                   self.max_depth,self.depth+1)
            self.right_child = Node(feature[feature[split_index] in split[1]],target[feature[split_index] in split[1]],
                                   self.max_depth,self.depth+1)

        def forward(features):
            if self.left_child is None or self.right_child is None:
                raise RuntimeError('You must call make_children first')
            if features[self.split_feature_index] in self.split[0]:
                self.left_child.forward(features[self.split_feature_index])
            elif features[self.split_feature_index] in self.split[1]:
                self.right_child.forward(features[self.split_feature_index])

class LeafNode:

    def __init__(self,feature,target):
        self.feature = feature
        self.target = target
        self.predicted_label = None

    def calculate_best_label(self):
        gini_impurities = []
        for i in range(len(self.feature)):
            gini_impurity_label, split = gini_impurity(self.feature,self.target)
            gini_impurities.append(gini_impurity_label)
        self.predicted_label = gini_impurities.index(max(gini_impurities))

    def forward(self,features):
        if self.predicted_label is not None:
            return self.predicted_label
        else:
            raise RuntimeError('You must call calculate_best_label first')