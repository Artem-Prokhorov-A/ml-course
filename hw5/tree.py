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
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ

    R = target_vector.shape[0]
    sorted_ids = np.argsort(feature_vector)
    feature_vector = feature_vector[sorted_ids]
    target_vector = target_vector[sorted_ids]

    counts = np.arange(1, R)
    prob_one_left = np.cumsum(target_vector[:-1]) / counts
    prob_one_right = (np.cumsum(target_vector[::-1][:-1]) / counts)[::-1]
    H_ls = 1 - (prob_one_left ** 2) - ((1 - prob_one_left) ** 2)
    H_rs = 1 - (prob_one_right ** 2) - ((1 - prob_one_right) ** 2)

    pre_ginis = -(counts / R * H_ls + counts[::-1] / R * H_rs)

    unique_features, cnt_vals = np.unique(feature_vector, return_counts=True)
    ids = (np.cumsum(cnt_vals) - 1)[:-1]

    ginis = pre_ginis[ids]
    id_gini_best = np.argmax(ginis)
    gini_best = ginis[id_gini_best]
    thresholds = (unique_features[1:] + unique_features[:-1]) / 2
    threshold_best = thresholds[id_gini_best]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree1:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self.depth = 1
        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def get_params(self, deep):
        return {'feature_types': self._feature_types}

    def _fit_node(self, sub_X, sub_y, node):
        if "depth" not in node.keys():
            node["depth"] = 1
        # changed `np.all(sub_y != sub_y[0])` to `np.all(sub_y == sub_y[0])`, added max_depth and _min_samples_split checks
        check_max_depth = (self._max_depth is not None and node["depth"] >= self._max_depth)
        check_min_samples_split = (self._min_samples_split is not None and sub_y.shape[0] < self._min_samples_split)
        if np.all(sub_y == sub_y[0]) or check_max_depth or check_min_samples_split:
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        # changed range
        print(sub_X)
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
                    # changed `current_count / current_click` to `current_click / current_count`
                    ratio[key] = current_click / current_count
                # changed `x[1]` to `x[0]`
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(np.unique(feature_vector)) == 1:
                continue

            thresholds, ginis, threshold, gini = find_best_split(feature_vector, sub_y)
            # added min_samples_leaf check
            print("глубина", node['depth'], "признак", feature, "холды", thresholds, "джины", ginis)


            left_sz = np.sum(feature_vector < threshold)
            right_sz = np.sum(feature_vector >= threshold)
            min_samples_leaf_is_ok = (self._min_samples_leaf is None or min(left_sz, right_sz) >= self._min_samples_leaf)
            if (gini_best is None or gini > gini_best) and min_samples_leaf_is_ok:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                # changed to lower case
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        # added depth calculations
        node["left_child"]["depth"] = node["depth"] + 1
        node["right_child"]["depth"] = node["depth"] + 1
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        # changed to sub_y[np.logical_not(split)]
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        real_split = (self._feature_types[feature] == 'real' and x[feature] < node["threshold"])
        categorical_split = (self._feature_types[feature] == 'categorical' and x[feature] in node["categories_split"])
        if real_split or categorical_split:
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        print('start_fit')
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

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