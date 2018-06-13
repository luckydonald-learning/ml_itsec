from math import sqrt

import numpy as np
import os


FOLDER = os.path.join('..','..', "Machine Learning for Computer Security", "Exercises", "mlsec-exer06-knn")

train_set = np.load(os.path.join(FOLDER, 'train.npy'))
test_set =  np.load(os.path.join(FOLDER, 'test.npy'))

LABELS = 0
DATA = 1
X = 0
Y = 1

class NearestNeighborClassification(object):
    def __init__(self, train_set=None, test_set=None, k=None):
        # train()
        if train_set is not None:
            self.train_set = train_set
        # end if

        if test_set is not None:
            self.test_set = test_set
        # end if
        self.k = k
    # end def

    def train(self):
        pass  # we already memorized the set :D
    # end def

    @staticmethod
    def _dataset_to_points(dataset):
        """
        Converts a dataset to (x,y) tuple.

        :param dataset: like `self.test_set` or `self.train_set`.
        :type  dataset: List[List[int], List[List[float],List[float]]]

        :return: generator, generating tuples with (x,y).
        :rtype: List[Tuple[float, float]]

        """
        # dataset = self.test_set
        yield from ((dataset[DATA][X][i], dataset[DATA][Y][i]) for i in range(len(dataset[LABELS])))
    # end def

    def test(self):
        X_set = list(self._dataset_to_points(self.train_set))
        self.evaluate(self.train_set[LABELS], X_set)
    # end def

    def evaluate(self, X=None, Y=None, k=None):
        """
        returning a list of boolean values indicating whether classify(x, k) returned the correct label y.

        :param X: a list of points to classify
        :type  X: list of tuple of float

        :param Y: list of labels of that points
        :type  Y: list of int

        :return: list of bool
        """
        for i in range(len(X)):
            detected = 1 if self.classify(X[i], k=k) else -1
            yield Y[i] == detected
        # end def
    # end def

    @staticmethod
    def _distance_between_points(x0, y0, x1, y1):
        dist = (sqrt(((x1 - x0) * (x1 - x0)) + ((y1 - y0) * (y1 - y0))))
        return (dist * 100) / 100
    # end def

    def classify(self, point, k=None, undecided_default=False):
        """
        Determine the k-nearest neighbors and return
        the most common label (in terms of frequency)
        among the k neighbors

        :param point: the point to classify
        :type  point: tuple of float

        :param x:  x = (x1, x2)
        :param k:  None: Use `self.k` set for this KNN. int: Use that k.

        :type  k: None|int
        :param undecided_default: Value to use, if

        :return: `True` if most of the nearest neighbors are labeled `1`, `False` if most are labeled `-1`,
        the value `undecided_default` if they are equally often present.
        :rtype: bool
        """
        liste = list()  # [distance, point_x, point_y, label]

        farest_k = None

        for i in range(len(self.train_set[LABELS])):
            label = self.train_set[LABELS][i]
            x1 = self.train_set[DATA][X][i]
            y1 = self.train_set[DATA][Y][i]
            distance = self._distance_between_points(point[0], point[1], x1, y1)
            if len(liste) < k:
                if farest_k is None:
                    farest_k = distance
                elif distance > farest_k:
                    # we store the farest away k,
                    # so when we fill the list up to k elements,
                    # the can just update that value.
                    farest_k = distance
                # end if
                liste.append([distance, x1, y1, label])
            else:
                if distance >= farest_k:
                    # skip if we already have k points which are nearer.
                    # or equally far away.
                    continue
                # end def

                # now we need to do somenting
                liste.append([distance, x1, y1, label])
                liste = sorted(liste, key=0)  # sort by distance
                del liste[-1]  # delete the farest away point
                farest_k = liste[-1][0]  # the new farest element.
            # end def
        # end for

        # liste now contains the k nearest elements.
        true_counter = 0
        false_counter = 0

        for distance, point_x, point_y, label in liste:
            if label == 1:
                true_counter += 1
            else:
                false_counter += 1
            # end if
        # end for

        if true_counter > false_counter:
            return True
        elif false_counter > true_counter:
            return False
        else:
            return undecided_default
        # end def
    # end def
# end def


if __name__ == '__main__':
    knn = NearestNeighborClassification(train_set=train_set, test_set=test_set)
    knn.train()  # lol
    knn.test()  # calls knn.evaluate(X,Y)