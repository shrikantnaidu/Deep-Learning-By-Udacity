import numpy as np

L=[5,6,7]
# Function that takes as input a list of numbers, and returns the list of values given by the softmax function.
def softmax(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result

Y=[1,0,1,1] 
P=[0.4,0.6,0.1,0.5]
# Function that takes as input two lists Y, P, and returns the float corresponding to their cross-entropy
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))