#%%
import numpy as np
import matplotlib.pyplot as plt


# %%
n = 100
Xtest = np.linspace(-5, 5, n).reshape(-1, 1)


def kernel(a, b, param):
    sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
    return np.exp(-0.5 * (1 / param) * sqdist)


# def kernel(a, b, param):
#     sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
#     return np.exp(-0.5 * (1 / param) * sqdist)


param = 0.1
K_ss = kernel(Xtest, Xtest, param)
# %%
L = np.linalg.cholesky(K_ss + 1e-6 * np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n, 3)))

plt.plot(Xtest, f_prior)
plt.axis([-5, 5, -3, 3])
plt.title("Three samples from the GP prior")
plt.show()
# %%
Xtrain = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
ytrain = np.sin(Xtrain)
#%%
K = kernel(Xtrain, Xtrain, param)
L = np.linalg.cholesky(K + 0.00005 * np.eye(len(Xtrain)))

# %%
K_s = kernel(Xtrain, Xtest, param)
Lk = np.linalg.solve(L, K_s)
mu = np.dot(Lk.T, np.linalg.solve(L, ytrain))


# %%n
s2 = np.diag(K_ss) - np.sum(Lk ** 2, axis=0)
stdv = np.sqrt(s2).reshape(-1, 1)
# %%
L = np.linalg.cholesky(K_ss + 1e-6 * np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1, 1) + np.dot(L, np.random.normal(size=(n, 3)))
# %%
plt.plot(Xtrain, ytrain, "bs", ms=8)
plt.plot(Xtest, f_post)
plt.gca().fill_between(
    Xtest.flat, (mu - 2 * stdv).flat, (mu + 2 * stdv).flat, color="#dddddd"
)
plt.plot(Xtest, mu, "r--", lw=2)

