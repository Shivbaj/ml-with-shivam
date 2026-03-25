from collections import defaultdict
import sys


# ---------------- NODE ----------------
class Node:

    def __init__(self, val):
        self.value = val              # feature index OR leaf label
        self.children = {}
        self.is_leaf = False
        self.majority = None

        # numeric support
        self.is_numeric = False
        self.threshold = None


# ---------------- DECISION TREE ----------------
class DecisionTree:

    def __init__(self, minimum_leaf, type_of_tree, max_depth=5):
        self.minimum_leaf = minimum_leaf
        self.type_tree = type_of_tree
        self.max_depth = max_depth
        self.root = None

    # ---------------- MAIN ----------------
    def make_tree(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    # ---------------- BUILD TREE ----------------
    def _build_tree(self, X, y, depth):

        # STOP CONDITION
        if len(set(y)) == 1 or len(y) <= self.minimum_leaf or depth >= self.max_depth:
            node = Node(self._majority(y))
            node.is_leaf = True
            node.majority = node.value
            return node

        min_gini = sys.maxsize
        best_feature_idx = None
        best_threshold = None
        best_is_numeric = False

        num_features = len(X[0])

        # ---------------- FIND BEST SPLIT ----------------
        for feature_idx in range(num_features):

            feature_data = [row[feature_idx] for row in X]

            # numeric feature
            if self._is_numeric(feature_data):

                thresholds = self._get_thresholds(feature_data)

                for t in thresholds:
                    gini = self._numeric_split_gini(X, y, feature_idx, t)

                    if gini < min_gini:
                        min_gini = gini
                        best_feature_idx = feature_idx
                        best_threshold = t
                        best_is_numeric = True

            # categorical feature
            else:
                gini = self._categorical_gini(feature_data, y)

                if gini < min_gini:
                    min_gini = gini
                    best_feature_idx = feature_idx
                    best_threshold = None
                    best_is_numeric = False

        # ---------------- CREATE NODE ----------------
        node = Node(best_feature_idx)
        node.majority = self._majority(y)
        node.is_numeric = best_is_numeric
        node.threshold = best_threshold

        # ---------------- SPLIT DATA ----------------
        if node.is_numeric:

            left_X, left_y = [], []
            right_X, right_y = [], []

            for i in range(len(X)):
                if X[i][best_feature_idx] <= node.threshold:
                    left_X.append(X[i])
                    left_y.append(y[i])
                else:
                    right_X.append(X[i])
                    right_y.append(y[i])

            node.children["left"] = self._build_tree(left_X, left_y, depth + 1)
            node.children["right"] = self._build_tree(right_X, right_y, depth + 1)

        else:

            splits = {}

            feature_column = [row[best_feature_idx] for row in X]

            for i, val in enumerate(feature_column):
                if val not in splits:
                    splits[val] = ([], [])

                splits[val][0].append(X[i])
                splits[val][1].append(y[i])

            for feature_value, (X_subset, y_subset) in splits.items():
                node.children[feature_value] = self._build_tree(
                    X_subset, y_subset, depth + 1
                )

        return node

    # ---------------- NUMERIC HELPERS ----------------
    def _is_numeric(self, column):
        return isinstance(column[0], (int, float))

    def _get_thresholds(self, column):
        sorted_vals = sorted(set(column))
        return [(sorted_vals[i] + sorted_vals[i+1]) / 2
                for i in range(len(sorted_vals) - 1)]

    def _numeric_split_gini(self, X, y, feature_idx, threshold):

        left, right = [], []

        for i in range(len(X)):
            if X[i][feature_idx] <= threshold:
                left.append(y[i])
            else:
                right.append(y[i])

        return self._weighted_gini(left, right)

    # ---------------- CATEGORICAL GINI ----------------
    def _categorical_gini(self, feature_column, y):

        feature_to_target = defaultdict(list)

        for f, label in zip(feature_column, y):
            feature_to_target[f].append(label)

        total = len(y)
        weighted = 0

        for group in feature_to_target.values():
            weighted += (len(group) / total) * self._gini(group)

        return weighted

    # ---------------- GINI CORE ----------------
    def _gini(self, group):
        from collections import defaultdict

        freq = defaultdict(int)
        for v in group:
            freq[v] += 1

        impurity = 1
        total = len(group)

        for c in freq.values():
            p = c / total
            impurity -= p ** 2

        return impurity

    def _weighted_gini(self, left, right):

        total = len(left) + len(right)

        return (
            (len(left) / total) * self._gini(left) +
            (len(right) / total) * self._gini(right)
        )

    # ---------------- MAJORITY ----------------
    def _majority(self, y):
        freq = defaultdict(int)
        for v in y:
            freq[v] += 1
        return max(freq, key=freq.get)

    # ---------------- PREDICT ----------------
    def predict_one(self, x):

        node = self.root

        while not node.is_leaf:

            feature_idx = node.value

            if node.is_numeric:

                if x[feature_idx] <= node.threshold:
                    node = node.children["left"]
                else:
                    node = node.children["right"]

            else:
                val = x[feature_idx]

                if val not in node.children:
                    return node.majority

                node = node.children[val]

        return node.value
    

def main():

    # ---------------- DATASET (categorical + numeric) ----------------
    X = [
        ["Sunny", 85],
        ["Sunny", 80],
        ["Overcast", 83],
        ["Rainy", 70],
        ["Rainy", 68],
        ["Rainy", 65],
        ["Overcast", 64],
        ["Sunny", 72],
        ["Sunny", 69],
        ["Rainy", 75],
        ["Sunny", 75],
        ["Overcast", 75],
        ["Overcast", 72],
        ["Rainy", 80]
    ]

    y = [
        "No", "No", "Yes", "Yes", "Yes",
        "No", "Yes", "No", "Yes", "Yes",
        "Yes", "Yes", "Yes", "No"
    ]

    # ---------------- TRAIN ----------------
    tree = DecisionTree(minimum_leaf=1, type_of_tree="gini", max_depth=3)
    tree.make_tree(X, y)

    # ---------------- TEST CASES ----------------
    test_samples = [
        ["Sunny", 85],
        ["Overcast", 72],
        ["Rainy", 70],
        ["Sunny", 65],
        ["Rainy", 90],   # unseen-ish numeric pattern
        ["Overcast", 80]
    ]

    # ---------------- PREDICT ----------------
    print("\n===== PREDICTIONS =====\n")

    for sample in test_samples:
        pred = tree.predict_one(sample)
        print(f"Input: {sample} -> Prediction: {pred}")


# ---------------- RUN ----------------
if __name__ == "__main__":
    main()