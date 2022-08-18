import solver
import matplotlib.pyplot as plt
import numpy as np
import time

domega = 2 * np.pi * 0.025
omega1 = 0
omega2 = 0
gamma1 = 2 * np.pi * 3
eta1 = 2 * np.pi * 0.1
gamma2 = 2 * np.pi * 3
alpha = 0.5

Evals = np.logspace(-2, 2, num=100)
Wvals = np.logspace(-2, 2, num=100)
orders = [-7, -5, -3, -1, 1, 3, 5, 7]

LEN = 10000

tic = time.perf_counter()
res = np.array(solver.calc(domega, omega1, omega2, gamma1,
    eta1, gamma2, alpha, Evals, Wvals, orders, LEN))
toc = time.perf_counter()
print(f"Runtime: {toc-tic:.4f} s")
np.save("result.npy", res)
# plt.figure()
# plt.plot(10*np.log10(np.abs(np.fft.fftshift(res))))
# plt.show()