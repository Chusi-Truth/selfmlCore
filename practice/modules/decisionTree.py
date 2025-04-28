import numpy as np

class DecisionTree():
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(sample, self.tree) for sample in X])

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # Stop criteria
        if len(unique_classes) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return unique_classes[0]

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.random.choice(unique_classes)

        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y):
        num_samples, num_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                gain = self._information_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, parent_y, left_y, right_y):
        weight_left = len(left_y) / len(parent_y)
        weight_right = len(right_y) / len(parent_y)
        
        gain = self._entropy(parent_y) - (weight_left * self._entropy(left_y) + weight_right * self._entropy(right_y))
        return gain
    
    def _entropy(self, y):
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _predict(self, sample, tree):
        if not isinstance(tree, tuple):
            return tree

        feature, threshold, left_subtree, right_subtree = tree
        if sample[feature] < threshold:
            return self._predict(sample, left_subtree)
        else:
            return self._predict(sample, right_subtree)

# 示例用法
if __name__ == "__main__":
    # 示例数据
    X = np.array([[2, 3], [1, 4], [3, 6], [2, 7], [3, 3], [1, 3]])
    y = np.array([0, 0, 1, 1, 0, 1])

    # 创建决策树实例并训练
    tree = DecisionTree(max_depth=3)
    tree.fit(X, y)

    # 预测
    predictions = tree.predict(X)
    print(f"Predictions: {predictions}")
