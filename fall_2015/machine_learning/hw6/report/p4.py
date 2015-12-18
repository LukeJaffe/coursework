import matplotlib.pyplot as plt

plt.plot([1,2,2], [1,2,0], 'rs', [0,1,0], [0,0,1], 'bs', [-1.0,2.5], [2.5,-1.0], 'g', [-1,2], [2,-1], 'y', [-1,3], [3,-1], 'y')
plt.axis([-1,3,-1,3])
plt.ylabel('y')
plt.xlabel('x')
plt.title('Problem 4')
plt.show()
