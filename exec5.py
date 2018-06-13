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


class Perceptron(object):
    @staticmethod
    def generate_random_tuple():
        return np.array([random.randint(0, 10) + random.random() for _ in range(2)])
    # end def

    def __init__(self, train_set=None, start_w=None, test_set=None):
        # train()
        self.history_changes = None
        self.history_w1 = []
        self.history_w0 = []
        if start_w:
            self.start_w = np.array(start_w)
        else:
            self.start_w = self.generate_random_tuple()
        # end if
        self.w = None
        if train_set is not None:
            self.train_set = train_set
        # end if

        # test()
        if test_set is not None:
            self.test_set = test_set
        # end if
        self.positives, self.negatives = None, None

        # self.balanced_error_rate()
        self._balanced_error_rate = None
    # end def

    def f(self, features):
        return f(self.w, features)
    # end def

    def train(self, learn_rate=1.0):
        if self.train_set is None:
            raise ValueError('no training set.')
        # end if

        self.w = self.start_w if self.w is None else self.w
        if self.w is None:
            raise ValueError('no (initial) weight vector set.')
        # end if

        N = len(self.train_set[LABELS][0])

        self.history_w0.append(self.w[0])
        self.history_w1.append(self.w[1])
        self.history_changes = 0 if self.history_changes is None else self.history_changes

        for i in range(N):
            ϕ = self.train_set[DATA][i]  # (x0, x1)
            prediction = self.f(ϕ)
            label = self.train_set[LABELS][0][i]  # label yi

            if prediction != label:  # label != prediction
                self.history_changes += 1
                self.w = self.w + learn_rate * label * ϕ
            # end if
            self.history_w0.append(self.w[0])
            self.history_w1.append(self.w[1])
        # end for
        return self.w
    # end def

    def train_multiple(self, max_count=10, learn_rate=1.0):
        change_count = self.history_changes
        self.train(learn_rate=learn_rate)
        for i in range(max_count-1):
            # train multiple times

            if learn_rate
            self.train(learn_rate=learn_rate)
            if change_count == self.history_changes:
                # no changes in one iteration
                break
            # end if
            change_count = self.history_changes
        # end for
    # end def

    def test(self):
        if self.test_set is None:
            raise ValueError('no test set.')
        # end if
        if self.w is None:
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
                self.negatives[label] += 1
            else:
                # is correct
                self.positives[label] += 1
            # end if
        # end for
        return self.positives, self.negatives
    # end def

    @property
    def balanced_error_rate(self):
        """
         Balanced Error Rate (BER)

        :raises ValueError: If something we depend on is not there.
        :return: float
        """
        if self._balanced_error_rate is None:
            if self.positives is None:
                raise ValueError('no positives calculated.')
                self.test()
            # end if
            if self.negatives is None:
                raise ValueError('no negatives calculated.')
                self.test()
            # end if
            self._balanced_error_rate = balanced_error_rate(self.positives, self.negatives)
        # end def
        return self._balanced_error_rate
    # end def

    def draw_training(self):
        in_pos = {'x': [], 'y': []}
        in_neg = {'x': [], 'y': []}
        if self.train_set is not None:
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
        # end if

        test_pos = {'x': [], 'y': []}
        test_neg = {'x': [], 'y': []}

        if self.test_set is not None:
            for i, element in enumerate(self.test_set[DATA]):
                x = element[0]
                y = element[1]
                if self.test_set[LABELS][0][i] == -1:
                    test_neg['x'].append(x)
                    test_neg['y'].append(y)
                else:
                    test_pos['x'].append(x)
                    test_pos['y'].append(y)
                # end if
            # end for
        # end if

        bg_pos = {'x': [], 'y': []}
        bg_neg = {'x': [], 'y': []}
        for x in range(-10, 90):
            for y in range(-10, 80):
                if f(self.w, [x, y]) == -1:
                    bg_neg['x'].append(x * 0.1)
                    bg_neg['y'].append(y * 0.1)
                else:
                    bg_pos['x'].append(x * 0.1)
                    bg_pos['y'].append(y * 0.1)
                # end if
            # end for
        # end for

        # generate label text
        text = ''
        if self.start_w is not None:
            for i in range(len(self.start_w)):
                text += "s{i} = {val!r}\n".format(i=i, val=self.start_w[i])
            # end for
        # end for
        if self.w is not None:
            for i in range(len(self.w)):
                text += "w{i} = {val!r}\n".format(i = i, val=self.w[i])
            # end for
        # end if
        if self.history_changes is not None:
            text += "w changes: {i}\n".format(i=self.history_changes)
        # end for

        try:
            text += 'BER = {ber}\n'.format(ber=self.balanced_error_rate)
        except ValueError:
            text = ''
        # end try
        text = None if len(text) == 0 else text.strip()

        print('lel')
        layout = GridSpec(3, 4)
        fig = plt.figure()

        subplt = fig.add_subplot(layout[0, :2])
        subplt.title.set_text('Training data')
        subplt.plot([_[0] for _ in self.train_set[DATA]], '.', label='x0')
        subplt.plot([_[1] for _ in self.train_set[DATA]], '.', label='x1')
        subplt.legend(loc="lower right")

        subplt = fig.add_subplot(layout[0, 2:])
        subplt.title.set_text('Weight updates')
        subplt.plot(self.history_w0, '.', label='w0')
        subplt.plot(self.history_w1, '.', label='w1')
        subplt.legend(loc="lower right")

        subplt = fig.add_subplot(layout[1:, :3])
        subplt.set_title('Feature Space')
        #subplt.plot(bg_pos['x'], bg_pos['y'], color=(0.7, 1, 0.7), marker='o', linestyle='', label='postive background')
        #subplt.plot(bg_neg['x'], bg_neg['y'], color=(1, 0.7, 0.7), marker='o', linestyle='', label='negative background')
        subplt.plot(test_pos['x'], test_pos['y'], color=(0, 1, 1), marker='.', linestyle='', label='postive test')
        subplt.plot(test_neg['x'], test_neg['y'], color=(1, 0, 1), marker='.', linestyle='', label='negative test')
        subplt.plot(in_pos['x'], in_pos['y'], color=(0.0, 1, 0.0), marker='.', linestyle='', label='postive learn')
        subplt.plot(in_neg['x'], in_neg['y'], color=(1.0, 0, 0.0), marker='.', linestyle='', label='negative learn')

        xmin, xmax = subplt.get_xlim()
        a = -self.w[0] / self.w[1]
        x_ = np.linspace(xmin, xmax)
        y_ = a * x_  # - (bias) / w[1]
        subplt.plot(x_, y_, label='hyperplane')

        legendplt = fig.add_subplot(layout[1:, 3:])
        # Put a legend to the right of the current axis
        legendplt.set_axis_off()
        legendplt.legend(*subplt.get_legend_handles_labels(), loc='center', title=text)
        legendplt.set_xlabel(text)

        return fig
    # end def

    def __lt__(self, other):
        """
        x<y calls x.__lt__(y)
        :param other:
        :return:
        """
        if isinstance(other, float):
            return self.balanced_error_rate > other
        # end if
        if not isinstance(other, self.__class__):
            raise ValueError('other is wrong type (neither {clazz} nor float)'.format(clazz=self.__class__.__name__))
        # end if

        if (
            self.balanced_error_rate == other.balanced_error_rate
            and self.history_changes is not None
            and other.history_changes is not None
        ):
            return self.history_changes > other.history_changes
        # end if
        return self.balanced_error_rate < other.balanced_error_rate
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
        [0.01, 4.56492131],
        [7.45923757, 7.02602091],
        [7.20224935, 3.38816243],
        [8.67041419, 2.38209922],
        [9.257037201559537, 0.3442845],
        #[2.639494716122672, -2.577715365455056],
    ]
    ps = []
    for i in range(len(randoms)):
        p = Perceptron(train_set=train_set, start_w=randoms[i], test_set=test_set)
        p.train(0.5)

        p.test()
        print('The one with weight {w!r} got a balanced error rate of {ber!r}.'.format(
            w=list(p.w), ber=p.balanced_error_rate
        ))
        ps.append(p)
    # end for

    for i in range(0):
        p = Perceptron(train_set=train_set, test_set=test_set)  # random start weight
        p.train()
        p.test()
        print('The one with weight {w!r} got a balanced error rate of {ber!r}.'.format(
            w=list(p.w), ber=p.balanced_error_rate
        ))
        ps.append(p)
    # end for

    # lowest rate is best rate.
    ps = list(sorted(ps, reverse=True))
    for p in ps[-1:]:  # only last one
        p.draw_training().show()
        pass
    # end for
    print('Best is the one with weight {w!r}, it got the lowest balanced error rate of {ber!r}'.format(w=list(ps[-1].w), ber=ps[-1].balanced_error_rate))
# end def


if __name__ == '__main__':
    print('yoooo')
    main()
# end if
