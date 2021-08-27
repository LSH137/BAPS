import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, w, b):
    return 1/(1 + np.exp(-x*w+b))


x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x, 0.5, 0)
y2 = sigmoid(x, 1, 0)
y3 = sigmoid(x, 2, 0)

plt.plot(x, y1, "r", linestyle='--')
plt.plot(x, y2, 'g')
plt.plot(x, y3, 'b', linestyle='--')
plt.plot([0, 0], [1.0, 0.0], ":")
plt.title("sigmoid function")
plt.show()
