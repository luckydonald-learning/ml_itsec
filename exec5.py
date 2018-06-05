import os
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


def balanced_error_rate(T, F):
    """
    Where Ti and Fi are the number of correctly and incorrectly
    classified data points from class i ∈ {−1, 1} respectively.
    :return:
    """
    return 0.5 * ((F[-1] / (F[-1] + T[-1])) + (F[1] / (F[1] + T[1])))
# end def


FOLDER = os.path.join('..','..', "Machine Learning for Computer Security", "Exercises", "mlsec-exer05-perceptron")

"""
⟨a, b⟩ = sum(ai*bi)
⟨a, b⟩ = sum(x0i*wi)
ϕ(row) = (x0, x1)
⟨ϕ(row), w⟩
f(row) = sign(⟨ϕ(row), w⟩) = sign(sum((x0, x1) * w) 
       = sign(sum((x0, x1) * w)
       = sign(sum(rows[i] * w[i] for i in range(N))
"""


print('lol')
train_set = np.load(os.path.join(FOLDER, 'train.npy'))
test_set =  np.load(os.path.join(FOLDER, 'test.npy'))
# [[-1 or +1], [2-tuple of floats, x0 and x1]]

"""
⟨a, b⟩ = sum(ai*bi)
⟨a, b⟩ = sum(x0i*wi)
ϕ(row) = (x0, x1)
⟨ϕ(row), w⟩
f(row) = sign(⟨ϕ(row), w⟩) = sign(sum((x0, x1) * w) 
       = sign(sum((x0, x1) * w)
       = sign(sum(rows[i] * w[i] for i in range(N))
       # oder?
       = sign(sum(feature * w for feature in [x0, x1])
"""


def f(w, features):
    return np.sign(sum(feature * w[_i] for _i, feature in enumerate(features)))  # prediction
# end def

LABELS = 0
DATA = 1


class Perception(object):
    @staticmethod
    def generate_random_tuple():
        return np.array([random.randint(0, 10) + random.random() for _ in range(2)])
    # end def

    def __init__(self, train_set=None, start_w=None, test_set=None):
        # train()
        self.history_w1 = []
        self.history_w0 = []
        if start_w:
            self.w = np.array(start_w)
        else:
            self.w = self.generate_random_tuple()
        # end if
        if train_set:
            self.train_set = train_set
        # end if

        # test()
        if test_set:
            self.test_set = test_set
        # end if
        self.positives, self.negatives = None, None

        # self.balanced_error_rate()
        self.error_rate = None
    # end def

    def f(self, features):
        return f(self.w, features)
    # end def

    def train(self):
        if not self.train_set:
            raise ValueError('no training set.')
        # end if
        N = len(self.train_set[LABELS][0])

        self.history_w1.append(self.w[0])
        self.history_w0.append(self.w[1])

        for i in range(N):
            ϕ = self.train_set[DATA][i]  # (x0, x1)
            prediction = self.f(ϕ)
            label = self.train_set[LABELS][0][i]  # label yi

            if prediction != label:  # label != prediction
                self.w = self.w + label * ϕ
            # end if
            self.history_w0.append(self.w[0])
            self.history_w1.append(self.w[1])
        # end for
        return self.w
    # end def

    def test(self):
        if not self.test_set:
            raise ValueError('no test set.')
        # end if
        if not self.w:
            raise ValueError('no weight vector.')
        # end if

        N = len(self.test_set[LABELS][0])

        self.negatives = {-1: 0, 1: 0}  # F: false, wrong detected.  Key: the assumed value
        self.positives = {-1: 0, 1: 0}  # T: true, correct detected. Key: the assumed value

        for i in range(N):
            ϕ = self.test_set[DATA][i]  # (x0, x1)
            prediction = self.f(ϕ)

            label = self.test_set[LABELS][0][i]  # label yi

            if prediction != label:
                # is wrong
                self.negatives[prediction] += 1
            else:
                # is correct
                self.positives[prediction] += 1
            # end if
        # end for
        return self.positives, self.negatives
    # end def

    def balanced_error_rate(self):
        """
         Balanced Error Rate (BER)
        :return: float
        """
        self.error_rate = balanced_error_rate(self.positives, self.negatives)
        return self.error_rate
    # end def

    def draw_training(self):
        in_pos = {'x': [], 'y': []}
        in_neg = {'x': [], 'y': []}

        for i, element in enumerate(self.train_set[DATA]):
            x = element[0]
            y = element[1]
            if self.train_set[LABELS][0][i] == -1:
                in_neg['x'].append(x)
                in_neg['y'].append(y)
            else:
                in_pos['x'].append(x)
                in_pos['y'].append(y)
            # end if
        # end for

        bg_pos = {'x': [], 'y': []}
        bg_neg = {'x': [], 'y': []}
        i = 0
        for x in range(0, 80):
            for y in range(0, 70):
                if f(self.w, [x, y]) == -1:
                    bg_neg['x'].append(x * 0.1)
                    bg_neg['y'].append(y * 0.1)
                else:
                    bg_pos['x'].append(x * 0.1)
                    bg_pos['y'].append(y * 0.1)
                # end if
            # end for
        # end for

        print('lel')
        layout = GridSpec(3, 2)
        fig = plt.figure()
        subplt = fig.add_subplot(layout[0, 0])
        subplt.title.set_text('Training data')
        subplt.plot([_[0] for _ in self.train_set[DATA]], '.', label='x0')
        subplt.plot([_[1] for _ in self.train_set[DATA]], '.', label='x1')
        subplt.legend(loc="lower right")

        subplt = fig.add_subplot(layout[0, 1])
        subplt.title.set_text('Weight updates')
        subplt.plot(self.history_w0, '.', label='w0')
        subplt.plot(self.history_w1, '.', label='w1')
        subplt.legend(loc="lower right")

        subplt = fig.add_subplot(layout[1:, :])
        subplt.title.set_text('Feature Space')
        subplt.plot(bg_pos['x'], bg_pos['y'], color=(0.7, 1, 0.7), marker='o', label='negative background')
        subplt.plot(bg_neg['x'], bg_neg['y'], color=(1, 0.7, 0.7), marker='o', label='postive background')
        subplt.plot(in_pos['x'], in_pos['y'], 'g.', label='postive')
        subplt.plot(in_neg['x'], in_neg['y'], 'r.', label='negative')
        subplt.legend(loc="upper right")
        plt.show()
    # end def

    def __lt__(self, other):
        """
        x<y calls x.__lt__(y)
        :param other:
        :return:
        """
        assert isinstance(other, Perception)
        assert other.error_rate is not None
        return self.error_rate > other.error_rate
    # end def
# end class


def main():
    # weights
    randoms = [
        [0.754645564, 0.087735],  # random, I hit my head on the keyboard.
        [4.42, 2.3],
        [6.79923014, 10.16876934],
        [4.13495359, 8.62620918],
        [5.20900703, 3.12870205],
        [4.49144523, 7.13199102],
        [0, 4.56492131],
        [7.45923757, 7.02602091],
        [7.20224935, 3.38816243],
        [8.67041419, 2.38209922]
    ]
    p = []
    winner = None
    for i in range(10):
        p.append(Perception(train_set=train_set, start_w=randoms[i], test_set=test_set))
        p[i].train()
        p[i].draw_training()
        print(p[i].balanced_error_rate())
        if winner is None or p[i] < winner:
            winner = p[i]
        # end if
        # lowest rate is best rate.
    # end for
    w = np.array([4.42, 2.3])  # random, I hit my head on the keyboard, another time


