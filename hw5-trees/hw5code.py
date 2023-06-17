import numpy as np
from collections import Counter

def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    threshold_best = 0
    gini_best = 1

    data = np.c_[feature_vector, target_vector]
    data = data[data[:, 0].argsort()]
    f, ids = np.unique(data[:, 0], return_index=True)
    f1 = f[:-1]
    f2 = f[1:]
    thresholds = (f1 + f2) / 2
    data = np.c_[feature_vector, target_vector]
    data = data[data[:, 0].argsort()]

    x = np.arange(1, feature_vector.shape[0] + 1)
    x_r = np.flip(x, axis=0) - 1
    left_1 = np.cumsum(data[:, 1])
    right_1 = data[:, 1][data[:, 1] == 1].shape[0] - np.cumsum(data[:, 1])
    left_0 = x - left_1
    right_0 = x_r - right_1

    left_0 = left_0[:-1]
    left_1 = left_1[:-1]
    right_0 = right_0[:-1]
    right_1 = right_1[:-1]
    x_r = x_r[:-1]
    x = x[:-1]

    H_right = 1 - (right_1/x_r) ** 2 - (right_0/x_r) ** 2
    H_left = 1 - (left_1/x) ** 2 - (left_0/x) ** 2
    ginis = -1 * (H_left * x / data.shape[0]) - (H_right * x_r / data.shape[0])
    #print(thresholds)
    ginis = ginis[ids[1:] - 1]
    id = np.argmax(ginis)
    gini_best = ginis[id]
    threshold_best = thresholds[id]
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._tree["depth"] = 1

        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    
    def _fit_node(self, sub_X, sub_y, node):
        pre_check = False
        if self._max_depth != None and node["depth"] >= self._max_depth:
            pre_check = True
        if self._min_samples_split != None and sub_X.shape[0] < self._min_samples_split:
            pre_check = True
        if np.all(sub_y == sub_y[0]) or pre_check:
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        #print(sub_X)
        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                feature_vector = np.array(np.fromiter(map(lambda x: categories_map[x], sub_X[:, feature]), 'int'))
            else:
                raise ValueError

            if len(np.unique(feature_vector)) == 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            # print("глубина", node['depth'], "признак", feature, "холды", thresholds, "джины", ginis)
            post_check = True
            if self._min_samples_leaf != None:
                if feature_vector[feature_vector < threshold].shape[0] < self._min_samples_leaf:
                    post_check = False
                if feature_vector[feature_vector >= threshold].shape[0] < self._min_samples_leaf:
                    post_check = False
            if (gini_best is None or gini > gini_best) and post_check:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            #node["class"] = Counter(sub_y).most_common(1)
            node["class"] = 0
            if sub_y[sub_y == 0].shape[0] < sub_y[sub_y == 1].shape[0]:
                node["class"] = 1
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"] = {"depth": node["depth"] + 1}
        node["right_child"] = {"depth": node["depth"] + 1}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        feature_best = node["feature_split"]
        if self._feature_types[feature_best] == "real":
            threshold = node["threshold"]
            if x[feature_best] < threshold:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            threshold = node["categories_split"]
            if x[feature_best] in threshold:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])


    def print_node(self, node):
        if node["type"] == "terminal":
            print(node["class"])
            return
        
        feature_best = node["feature_split"]
        if self._feature_types[feature_best] == "real":
            threshold = node["threshold"]
            print("Real", feature_best, threshold)
            print("left:")
            self._predict_node(node["left_child"])
            print("left:")
            self._predict_node(node["right_child"])
        else:
            threshold = node["threshold"]
            print("Categorial", feature_best, threshold)
            print("left:")
            self._predict_node(node["left_child"])
            print("left:")
            self._predict_node(node["right_child"])


    def print_tree(self):
        self.print_node(self._tree)


    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
