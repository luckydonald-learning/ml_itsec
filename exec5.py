import os
import numpy as np


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

LABELS = 0
DATA = 1

w = np.array([0.754645564, 0.087735])  # random, I hit my head on the keyboard.
# weights

N = len(train_set[LABELS][0])


for i in range(N):
    x0, x1 = train_set[DATA][i]  # (x0, x1)
    ϕ = np.array([x0, x1])  # feature vector

    f = np.sign(sum(feature * w for feature in [x0, x1]))  # prediction

    yi = train_set[LABELS][0][i]  # label

    if f != yi:  # label != prediction
        w = w + yi * ϕ