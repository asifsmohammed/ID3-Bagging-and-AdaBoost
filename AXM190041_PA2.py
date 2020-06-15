# decision_tree.py

#Submitted By:
#Asif Sohail Mohammed - AXM190041
#Pragya Nagpal - PXN190012

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math

from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    values, counts = np.unique(x, return_counts=True)
    return dict(zip(values, counts))
    
def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    z = partition(y)
    e = 0
    for key in z:
        e += -z.get(key)/len(y) * np.log2(z.get(key)/len(y))
    return e
    
def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    e = entropy(y)
    e_temp = 0
    not_x=[]
    for i in range(len(y)):
        if i not in x:
            not_x.append(i)
    e_temp = len(x) / len(y) * entropy(y[x]) + (len(not_x)/len(y))*entropy(y[not_x])
    return e - e_temp


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.
    
    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, 0True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    attribute_value_pairs = []
    for i in range(len(x[0])):
        values = np.unique(x[:,i])
        for j in values:
            tup = (i, j)
            attribute_value_pairs.append(tup)

    # Condition 1: all y values are same
    z = partition(y)
    if(len(z) == 1):
        return list(z.keys())[0]
    
    # Condition 2
    elif(len(attribute_value_pairs) == 0):
        maximum = max(z.values())
        for key, value in z.items():
            if value == maximum:
                return key
                
    # Condition 3
    elif(max_depth == depth):
        maximum = max(z.values())
        for key, value in z.items():
            if value == maximum:
                return key

    else:
        max_ig = 0
        for i in range(len(x[0])):
            values = np.unique(x[:,i])
            for j in values:
                ig = mutual_information(np.where(x[:, i] == j)[0],y)
                # ig = mutual_information(np.extract(x[:, i] == j, x[:, i]), y)
                # print("For Tuple (%d, %d) Information Gain is %f" % (i , j, ig))
                if(ig > max_ig):
                    max_ig = ig
                    max_ig_tup = (i, j)
        attribute_value_pairs.remove(max_ig_tup)
        
        # Recursive call
        return {(max_ig_tup[0], max_ig_tup[1], True) : id3(x[np.where(x[:, max_ig_tup[0]] == max_ig_tup[1])], y[np.where(x[:, max_ig_tup[0]] == max_ig_tup[1])], 
                                attribute_value_pairs, depth+1, max_depth),
        (max_ig_tup[0], max_ig_tup[1], False) : id3(x[np.where(x[:, max_ig_tup[0]] != max_ig_tup[1])], y[np.where(x[:, max_ig_tup[0]] != max_ig_tup[1])], 
                    attribute_value_pairs, depth+1, max_depth)}
        
    # raise Exception('Function not yet implemented!')

def predict_eg(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    """
    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    for key, value in tree.items():
        if((x[key[0]] == key[1]) == key[2]):
            if(isinstance(value, dict)):
                return(predict_example(x, value))
            else:
                return value
    # raise Exception('Function not yet implemented!')
    """
    if type(tree) is not dict: #terminate conditions if leaf node is reached
        return tree
    else:
        for i,j in tree.items():
            if((i[1]==x[i[0]])==i[2]):
                return predict_eg(x,tree[i])

def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    count =  0
    for i in range(len(y_pred)):
        if(y_true[i] != y_pred[i]):
            count += 1
    return (1 / len(y_true)) * count
    raise Exception('Function not yet implemented!')


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

def conf_matrix(ytst, y_pred):
    # Confusion Matrix
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(ytst)):
        if(ytst[i] == 1 and y_pred[i] == 1):
            tp += 1
        elif(ytst[i] == 0 and y_pred[i] == 0):
            tn += 1
        elif(ytst[i] == 0 and y_pred[i] == 1):
            fp += 1
        else:
            fn += 1
    cm = np.zeros( (2, 2), dtype=np.int32 )
    cm[0][0] = tp
    cm[0][1] = fn
    cm[1][0] = fp
    cm[1][1] = tn
    return cm

# Takes x, y, max_depth, num_trees and returns y_pred
def bagging(x, y, max_depth, num_trees):
    hypothesis = []

    for i in range(1, num_trees + 1):
        # Creating bootstraps
        M = subsample(x, y)

        ytrn_bs = M[:, 0]
        Xtrn_bs = M[:, 1:]

        decision_tree = (1,id3(Xtrn_bs, ytrn_bs, max_depth=max_depth))
        hypothesis.append(decision_tree)

    return hypothesis


# Create a random subsample from the dataset with replacement
def subsample(x, y):
	sample = []
	sample_size = len(x)
	while len(sample) < sample_size:
		# random.seed(len(sample))
		index = random.randint(0, sample_size - 1)
		sample.append(np.insert(x[index], 0, y[index]))
	return np.array(sample)

# Returns the dataset for next hypothesis and it samples according to weight of each example
def sample_boosting(x,y,w):
    X=[]
    Y=[]
    cw=np.cumsum(w)
    for i in range(len(cw)):
        rn=np.random.uniform(0.0,1.0)
        for j in range(len(cw)): 
            if rn>cw[j-1] and rn<=cw[j]:
                X.append(x[i])
                Y.append(y[i])
                break
    return np.array(X),np.array(Y)
    
# Takes x, y, maximum depth and number of stumps and returns h_ens
def boosting(x,y, max_depth, bag_size):
    h_ens=[0]*bag_size
    w = [1/len(x)]*len(x)
    w1=[0]*len(w)
    for i in range(bag_size):
    
        w = [1/len(x)]*len(x)
        hypothesis=id3(x, y, max_depth=max_depth)
        y_pred = [predict_eg(u, hypothesis) for u in x]
        error = compute_error(y, y_pred)
        alpha=0.5*np.log((1-error)/error)
        h_ens[i]=(alpha,hypothesis)
        
        #weights again
        for j in range(len(w)):
            if y[j]==y_pred[j]:
                w1[j]=w[j]*math.exp(-alpha)
            else:
                w1[j]=w[j]*math.exp(alpha)
        #normalizing
        w1=w1/np.sum(w1)
        x,y=sample_boosting(x,y,w)
        
    return h_ens
    
# Takes x, h_ens and returns predictions
def predict_example(x,h_ens):
    y_ens = {}
    for h in h_ens:
        y = predict_eg(x,h[1]) 
        if y in y_ens:
            y_ens[y] += h[0]
        else:
            y_ens[y] = h[0]
    return max(y_ens, key=lambda key: y_ens[key])

if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./data/mushroom.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    Xtrn = M[:, 1:]
    ytrn = M[:, 0]
    
    # Load the test data
    M = np.genfromtxt('./data/mushroom.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    Xtst = M[:, 1:]
    ytst = M[:, 0]
     
    # Bagging
    print("Bagging")
    d = [3, 5]
    k = [10, 20]

    for max_depth in d:
        for bag_size in k:
            h_ens= bagging(Xtrn, ytrn, max_depth, bag_size)

            # Training
            y_pred=[predict_example(x,h_ens) for x in Xtrn]
            trn_err=compute_error(ytrn,y_pred)
            
            y_pred=[predict_example(x,h_ens) for x in Xtst]
            tst_err=compute_error(ytst,y_pred)

            print("Max depth is {0} Bag size is {1}".format(max_depth, bag_size))
            print('Train Error is {0:4.2f}%.'.format(trn_err * 100))
            print('Test Error is {0:4.2f}%.'.format(tst_err * 100))
            print("Confusion Matrix")
            print(conf_matrix(ytst, y_pred))
      
    #Boosting
    d = [1, 2]
    k = [20, 40]
    
    print("Boosting")
    for max_depth in d:
        for bag_size in k:
            h_ens=boosting(Xtrn,ytrn,max_depth,bag_size)
            y_pred=[predict_example(x,h_ens) for x in Xtrn]
            trn_err=compute_error(ytrn,y_pred)
            
            y_pred=[predict_example(x,h_ens) for x in Xtst]
            tst_err=compute_error(ytst,y_pred)
            
            print("Max depth is {0} Bag size is {1}".format(max_depth, bag_size))
            print('Train Error is {0:4.2f}%.'.format(trn_err * 100))
            print('Test Error is {0:4.2f}%.'.format(tst_err * 100))
            print("Confusion Matrix")
            print(conf_matrix(ytst, y_pred))
    
    # Scikit-learn
    # Bagging
    print("Scikit-learn Bagging")
    d = [3, 5]
    k = [10, 20]

    def bagging_sklearn(x, y, max_depth, num_trees):
        model = BaggingClassifier(DecisionTreeClassifier(max_depth=max_depth), n_estimators=num_trees)
        model.fit(x, y)
        return model

    for max_depth in d:
        for bag_size in k:
            model = bagging_sklearn(Xtrn, ytrn, max_depth, bag_size)
            y_pred = model.predict(Xtst)
            print("Max depth is {0} Bag size is {1}".format(max_depth, bag_size))
            print("Train accuracy is {0}".format(model.score(Xtrn, ytrn)))
            print("Test accuracy is {0}".format(model.score(Xtst, ytst)))
            print("Confusion Matrix")
            print(confusion_matrix(ytst, y_pred))
    
    # Boosting
    print("Scikit-learn Boosting")
    d = [1, 2]
    k = [20, 40]

    def boosting_sklearn(x, y, max_depth, num_stumps):
        model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth), n_estimators=num_stumps)
        model.fit(x, y)
        return model

    for max_depth in d:
        for bag_size in k:
            model = bagging_sklearn(Xtrn, ytrn, max_depth, bag_size)
            y_pred = model.predict(Xtst)
            print("Max depth is {0} Bag size is {1}".format(max_depth, bag_size))
            print("Train accuracy is {0}".format(model.score(Xtrn, ytrn)))
            print("Test accuracy is {0}".format(model.score(Xtst, ytst)))
            print("Confusion Matrix")
            print(confusion_matrix(ytst, y_pred))