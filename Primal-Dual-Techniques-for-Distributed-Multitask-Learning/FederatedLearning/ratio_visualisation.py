import matplotlib.pyplot as plt

node = [2, 3, 5, 7, 10, 11, 12, 60, 100]
ratio = [0.25, 0.25, 0.22, 0.25510204081632654, 0.2125, 0.2190082644628099, 0.2326388888888889, 0.25625, 0.257125]

plt.plot(node, ratio)
plt.xlabel('number of nodes per block')
plt.ylabel('ratio')
plt.show()
plt.close()