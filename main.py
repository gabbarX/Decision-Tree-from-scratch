from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np


# Define the node class
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


# Define Decision Tree class
class DecisionTree:
    # Define initial inputs
    def __init__(self, max_depth=100, min_splits=20):
        self.minSplits = min_splits
        self.max_depth = max_depth

    # Define the fit function
    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.root = self.grow(X, y)

    # Define the stopping criteria
    def canStop(self, depth, X):
        samples = X.shape[0]
        labels = len(np.unique(y))
        return depth >= self.max_depth or labels == 1 or samples < self.minSplits

    # Define the best split function
    def selectRandomFeatures(self, X):
        feats = X.shape[1]
        feature_indices = np.random.permutation(feats)[: self.n_features]
        return feature_indices

    # Define the grow function
    def grow(self, X, y, depth=0):
        if self.canStop(depth, X):
            unique, counts = np.unique(y, return_counts=True)
            most_common_idx = np.argmax(counts)
            val = unique[most_common_idx]
            return Node(value=val)

        featureIdx = self.selectRandomFeatures(X)
        # find the best split
        bestFeature, best_thresh = self.bestSplit(X, y, featureIdx)

        # create child nodes
        leftIdx = np.argwhere(X[:, bestFeature] <= best_thresh).flatten()
        rightIdx = np.argwhere(X[:, bestFeature] > best_thresh).flatten()
        left = self.grow(X[leftIdx, :], y[leftIdx], depth + 1)
        right = self.grow(X[rightIdx, :], y[rightIdx], depth + 1)
        return Node(bestFeature, best_thresh, left, right)

    def CalculateEntropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        entropy = []
        for x in ps:
            if x > 0:
                entropy.append(x * np.log(x))
        return -np.sum(entropy)

    def getThreshold(self, X, feat_idx):
        # Get the unique values of the feature
        unique_values = np.sort(np.unique(X[:, feat_idx]))

        # Compute midpoints between adjacent unique values as thresholds
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2

        return thresholds

    def bestSplit(self, X, y, featureIdx):
        best_gain = -1
        split_idx = -1
        split_threshold = -1

        for feat_idx in featureIdx:
            thresholds = self.getThreshold(X, feat_idx)

            for thr in thresholds:
                # calculate the information gain
                # parent entropy
                parent_entropy = self.CalculateEntropy(y)

                # create children
                leftIdx = np.argwhere(X[:, feat_idx] <= thr).flatten()
                rightIdx = np.argwhere(X[:, feat_idx] > thr).flatten()

                # calculate the weighted avg. entropy of children
                n = len(y)
                nLeft = len(leftIdx)
                nRight = len(rightIdx)
                entropyLeft = self.CalculateEntropy(y[leftIdx])
                entropyRight = self.CalculateEntropy(y[rightIdx])
                child_entropy = (nLeft / n) * entropyLeft + (nRight / n) * entropyRight

                # calculate the Information Gain
                gain = parent_entropy - child_entropy
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def predict(self, X):
        predictions = [self.moveInTree(x, self.root) for x in X]
        return np.array(predictions)

    def moveInTree(self, x, node):
        if node.value != None:
            return node.value

        if x[node.feature] > node.threshold:
            return self.moveInTree(x, node.right)
        return self.moveInTree(x, node.left)

    def calculateAccuracy(self, Y_test, Y_pred):
        accuracy = np.sum(Y_test == Y_pred) / len(Y_test)
        return accuracy


data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = DecisionTree(max_depth=2, min_splits=20)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


acc = clf.calculateAccuracy(y_test, predictions)
print(acc)
