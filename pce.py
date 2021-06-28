#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
x = 4*np.random.rand(int(1e7))
plt.hist(x)
y = x ** 3 - 1
# %%
plt.hist(y)
# %%
y.mean()
# %%
import chaospy
# %%
q0,q1 =chaospy.variable(2)
# %%
q0
# %%
chaospy.monomial(5)
# %%
chaospy.monomial(2, 3, dimensions=2, cross_truncation=np.inf)
# %%
q0**np.arange(5)
# %%
2**np.arange(3)
# %%
