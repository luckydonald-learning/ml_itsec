import os
import numpy as np
from numpy import linalg as lala
import pylab
from PIL import Image
import matplotlib.pyplot as plt


FOLDER = os.path.join('..','..', "Machine Learning for Computer Security", "Exercises", "mlsec-exer07-pca", "images")

data = []
mean = np.zeros((1,676))
counter = 0
for filename in os.listdir(FOLDER):
    # print(filename)
    file = os.path.join(FOLDER, filename)
    pixels = []
    with Image.open(file) as image:
        for x in range(image.size[0]):
            for y in range(image.size[1]):
                if image.mode == 'L':
                    b = image.getpixel((y,x))  # black channel only
                    pixels.append(b)
                else:
                    r, g, b = image.getpixel((y,x))  # red, green, blue channels
                    pixels.append(r)
                # end if
            # end for
        # end for
    # end with
    img = np.array(pixels)
    mean += img
    counter += 1
    data.append(img)
# end for
mean /= counter  # calculating average

plt.imshow(np.resize(data[0], (26, 26)), cmap=plt.cm.gray)
plt.title('original image #1')
plt.show()

plt.imshow(np.resize(mean, (26, 26)), cmap=plt.cm.gray)
plt.title('mean')
plt.show()

diff = np.zeros((213, 676))
for i in range(len(data)):
    diff[i] = data[i].astype('float64') - mean
# end for

plt.imshow(np.resize(diff[0], (26, 26)), cmap=plt.cm.gray)
plt.title('diff image #1')
plt.show()

layout = pylab.GridSpec(213+1, 4)
fig = plt.figure()
subplt = fig.add_subplot(layout[0, 0])
subplt.set_title('#')
subplt = fig.add_subplot(layout[0, 1])
subplt.set_title('o')
subplt = fig.add_subplot(layout[0, 2])
subplt.set_title('m')
subplt = fig.add_subplot(layout[0, 3])
subplt.set_title('d')


for i in range(len(data))[::50]:
    subplt = fig.add_subplot(layout[i + 1, 0])
    subplt.set_title('#{}'.format(i))

    subplt = fig.add_subplot(layout[i+1, 1])
    subplt.imshow(np.resize(data[i], (26, 26)), cmap=plt.cm.gray)


    subplt = fig.add_subplot(layout[i+1, 2])
    subplt.imshow(np.resize(mean, (26, 26)), cmap=plt.cm.gray)

    subplt = fig.add_subplot(layout[i+1, 3])
    subplt.imshow(np.resize(diff[i], (26, 26)), cmap=plt.cm.gray)
# end def
plt.show()

convergence = np.cov(diff.T)

plt.imshow(np.resize(convergence[0], (26, 26)), cmap=plt.cm.gray)
plt.title('convergence #{}'.format(1))
plt.show()


eigenvalue, eigenvector = lala.eigh(convergence)  # Hermitian or symmetric matrix. The latter is true.


for i in range(len(eigenvector))[:3]:
    plt.imshow(np.resize(eigenvector[i], (26, 26)), cmap=plt.cm.gray)
    plt.title('eigenvector #{}'.format(i))
    plt.show()