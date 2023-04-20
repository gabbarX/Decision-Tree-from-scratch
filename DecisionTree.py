import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def isLeaf(self):
        if self.value == None:
            return False
        else:
            return True


class DecisionTree:
    def __init__(self, min_samplessplitter=2, max_depth=100, n_features=None):
        self.min_samplessplitter = min_samplessplitter
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        if self.n_features == None:
            self.n_features = X.shape[1]
        else:
            self.n_features = min(X.shape[1], self.n_features)
        self.root = self.growTree(X, y)

    def growTree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samplessplitter
        ):
            leaf_value = self.labelCounter(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature, best_thresh = self.bestSplit(X, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self.splitter(X[:, best_feature], best_thresh)
        left = self.growTree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self.growTree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def bestSplit(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self.infoGain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def infoGain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self.CalculateEntropy(y)

        # create children
        left_idxs, right_idxs = self.splitter(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self.CalculateEntropy(y[left_idxs]), self.CalculateEntropy(
            y[right_idxs]
        )
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def splitter(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def CalculateEntropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def labelCounter(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self.traverseDecisionTree(x, self.root) for x in X])

    def traverseDecisionTree(self, x, node):
        if node.isLeaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverseDecisionTree(x, node.left)
        return self.traverseDecisionTree(x, node.right)
