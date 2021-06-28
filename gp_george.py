# %%
import george
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score

#%%
n_data = 15
x = 10 * np.sort(np.random.rand(n_data))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))

# plt.errorbar(x, y, yerr=0.2, fmt=".k", capsize=0)
# %%
kernel = np.var(y) * george.kernels.ExpSquaredKernel(0.5)
gp = george.GP(kernel)
gp.compute(x, yerr)

x_pred = np.linspace(0, 10, 500)
pred, pred_var = gp.predict(y, x_pred, return_var=True)

plt.fill_between(
    x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var), color="k", alpha=0.2
)
plt.plot(x_pred, pred, "k")
plt.plot(x_pred, np.sin(x_pred), "--g")
plt.errorbar(x, y, yerr=0.2, fmt=".k", capsize=0)
plt.xlabel("x")
plt.ylabel("y")

#%%
print(gp.kernel)
print(f"Initial log-likelihood:{gp.log_likelihood(y)}")


def neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.log_likelihood(y)


def grad_neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y)


result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
# print(result)

gp.set_parameter_vector(result.x)
print(f"Final log-likelihood: {gp.log_likelihood(y)}")
print(result.x, gp.kernel)
# %%
pred, pred_var = gp.predict(y, x_pred, return_var=True)

plt.fill_between(
    x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var), color="k", alpha=0.2
)
plt.plot(x_pred, pred, "k", lw=1.5, alpha=0.5)
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x_pred, np.sin(x_pred), "--g")
plt.xlim(0, 10)
plt.ylim(-1.45, 1.45)
plt.xlabel("x")



# %%
