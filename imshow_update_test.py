import numpy as np
import matplotlib.pyplot as plt

image = np.random.randn(10,10)

plt.ion()

plt.figure(1)
im = plt.imshow(image)


for i in range(10):
    print i
    image = np.random.randn(10,10)
    im.set_data(image)
    plt.draw()
    plt.pause(1)
