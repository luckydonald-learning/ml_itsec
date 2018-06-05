import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


def balanced_error_rate(F, T):
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



def absum(N, a, b):
    result = 0.0
    for i in range(N):
        result += a[i] * b[i]
    # end for
    return result

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

w = np.array([0.754645564, 0.087735])  # random, I hit my head on the keyboard.
w = np.array([4.42, 2.3])  # random, I hit my head on the keyboard.
# weights

N = len(train_set[LABELS][0])

w0 = [w[0]]
w1 = [w[1]]

for i in range(N):
    x0, x1 = train_set[DATA][i]  # (x0, x1)
    ϕ = np.array([x0, x1])  # feature

    prediction = f(w, ϕ)

    label = train_set[LABELS][0][i]  # label yi

    if prediction != label:  # label != prediction
        w = w + label * ϕ
    # end if
    w0.append(w[0])
    w1.append(w[1])
# end for


pos = {'x': [], 'y': []}
neg = {'x': [], 'y': []}

for i, element in enumerate(train_set[DATA]):
    x = element[0]
    y = element[1]
    if train_set[LABELS][0][i] == -1:
        neg['x'].append(x)
        neg['y'].append(y)
    else:
        pos['x'].append(x)
        pos['y'].append(y)
    # end if
# end for

print('lel')
layout = GridSpec(3,2)
fig = plt.figure()
subplt = fig.add_subplot(layout[0, 0])
subplt.title.set_text('Training data')
subplt.plot([_[0] for _ in train_set[DATA]], '.', label='x0')
subplt.plot([_[1] for _ in train_set[DATA]], '.', label='x1')
subplt.legend(loc="lower right")


subplt = fig.add_subplot(layout[0, 1])
subplt.title.set_text('Weight updates')
subplt.plot(w0, '.', label='w0')
subplt.plot(w1, '.', label='w1')
subplt.legend(loc="lower right")


subplt = fig.add_subplot(layout[1:, :])
subplt.title.set_text('Feature Space')
subplt.plot(pos['x'], pos['y'], 'g.', label='postive')
subplt.plot(neg['x'], neg['y'], 'r.', label='negative')

subplt.plot([0, w[0]*2], [0, w[1]*-2], label='w')
subplt.legend(loc="lower right")


print('lel')
layout = GridSpec(3,2)
fig = plt.figure()
subplt = fig.add_subplot(layout[1:, :])
pos = {'x': [], 'y': []}
neg = {'x': [], 'y': []}
i = 0
for x in range(0, 70):
    for y in range(0, 70):
        if f(w, [x,y]) == -1:
            neg['x'].append(x*0.1)
            neg['y'].append(y*0.1)
        else:
            pos['x'].append(x*0.1)
            pos['y'].append(y*0.1)
        # end if
    # end for
# end for
subplt.plot(pos['x'], pos['y'], 'g.', label='postive')
subplt.plot(neg['x'], neg['y'], 'r.', label='negative')


plt.show()