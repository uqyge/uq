#%%
import numpy as np
import matplotlib.pyplot as pl

np.random.seed(1234)
x = np.sort(np.random.uniform(0, 10, 50000))
yerr = 0.1 * np.ones_like(x)
y = np.sin(x)
# %%
import george
from george import kernels
kernel = np.var(y) * kernels.ExpSquaredKernel(1.0)

gp_basic = george.GP(kernel)
gp_basic.compute(x[:100], yerr[:100])
print(gp_basic.log_likelihood(y[:100]))
# %%
gp_hodlr = george.GP(kernel, solver=george.HODLRSolver, seed=42)
gp_hodlr.compute(x[:100], yerr[:100])
print(gp_hodlr.log_likelihood(y[:100]))
# %%
