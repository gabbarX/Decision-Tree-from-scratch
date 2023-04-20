import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pprint import pprint
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the data
iris = load_iris()
X = iris.data
y = iris.target


# checks if the data is pure or not. Data is pure if it contains only 1 class
def isPure(y):
    unique = np.unique(y)
    if len(unique) == 1:
        return True
    else:
        return False


print(isPure([0, 0, 0, 0, 0]))
