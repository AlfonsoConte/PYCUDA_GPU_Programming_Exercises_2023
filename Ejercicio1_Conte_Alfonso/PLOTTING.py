from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

x = [400, 1000, 1400, 2000]
y = [[0.111926, 1.71413, 4.68423, 13.6762],
     [0.449085, 2.83652, 5.66883, 11.8227], [
         0.249231, 0.604308, 0.826312, 1.15677],
     [0.479743, 3.20425, 6.1839, 12.7412], [0.23982, 0.565538, 0.785574, 1.11208]]
labels = ['TCPU', 'TGPU1D', 'SGPU1D', 'TGPU2D', 'SGPU2D']

for y_arr, label in zip(y, labels, ):
    plt.grid()
    plt.xlabel('N values')
    plt.ylabel('Time(sec)')
    plt.plot(x, y_arr, marker="o", markersize=10, label=label)
plt.title("B=64")
plt.legend()
plt.show()


y = [[0.123991, 1.71402, 4.83138, 13.6567],
     [0.452329, 2.93872, 5.7191, 11.7164], [
         0.274117, 0.583253, 0.844779, 1.16561],
     [0.497601, 3.12705, 6.22792, 13.5559], [0.230787, 0.567247, 0.779875, 1.0451]]
labels = ['TCPU', 'TGPU1D', 'SGPU1D', 'TGPU2D', 'SGPU2D']

for y_arr, label in zip(y, labels, ):
    plt.grid()
    plt.xlabel('N values')
    plt.ylabel('Time(sec)')
    plt.plot(x, y_arr, marker="o", markersize=10, label=label)
plt.title("B=256")
plt.legend()
plt.show()


y = [[0.11734, 1.7654, 4.68884, 13.7209],
     [0.497182, 2.85916, 5.72187, 11.6246], [0.23601, 0.617455, 0.81946, 1.18033
                                             ],
     [0.440387, 3.16693, 6.52342, 14.3608], [0.261027, 0.56316, 0.750689, 0.98721]]
labels = ['TCPU', 'TGPU1D', 'SGPU1D', 'TGPU2D', 'SGPU2D']

for y_arr, label in zip(y, labels, ):
    plt.grid()
    plt.xlabel('N values')
    plt.ylabel('Time(sec)')
    plt.plot(x, y_arr, marker="o", markersize=10, label=label)
plt.title("B=1024")
plt.legend()
plt.show()
