# %%
import george
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyDOE import lhs
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

df_pred_org = pd.read_excel("data/9-10五维预测用100000.xlsx")
#%%
df_train = pd.read_excel("data/1000训练用.xlsx").sample(frac=1)
# df_train = df_pred_org.sample(1_000)
df_pred = df_pred_org.sample(4_000)
# df_pred = df_train.sample(800)

scaler = MinMaxScaler()
#%%
%%time
opt = True
n_trials = 20

input_label = ["A14", "A23", "Tup", "Tmid", "Tdown"]
# input_label = ["A1", "A2", "A3", "A4"]
output_label = "23z"

x = df_train[input_label].values
x = scaler.fit_transform(x)
yerr = 0 * np.ones(len(x))
y = df_train[output_label].values
x_pred = df_pred[input_label].values
x_pred = scaler.transform(x_pred)

# lb = np.array([1e-8, 0])
# ub = np.array([1, 1e8])
# lb = np.array([1e-1,1e5])
# ub = np.array([1,1e7])
lb = np.array([1e-1, 0])
ub = np.array([10, 1e2])


trials = lb + (ub - lb) * lhs(len(lb), n_trials).round(4)

r2s = []
lkhs = []
r2s_opt = []
lkhs_opt = []
hyper_opt = []
gps = []

for i in trials:
    print(i)
    a, b = i

    # kernel = np.var(y) * george.kernels.ExpSquaredKernel(100, ndim=x.shape[1])
    kernel = a * george.kernels.ExpSquaredKernel(b, ndim=x.shape[1])

    # gp = george.GP(kernel,solver=george.HODLRSolver,seed=42)
    gp = george.GP(kernel)
    gp.compute(x)

    pred, pred_var = gp.predict(y, x_pred, return_var=True)
    r2_pred = r2_score(df_pred[output_label], pred)

    # print(f'log_likelihood:{gp.log_likelihood(y)}')
    # print(f'r2:{r2_pred}')
    r2s.append(r2_pred)
    lkhs.append(gp.log_likelihood(y))
    # plt.plot(df_pred[output_label], pred, "d")
    # plt.title(f'r2={r2_pred}')

    # print(f"Initial log-likelihood:{gp.log_likelihood(y)}")

    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y, quiet=True)

    if opt == True:
        try:
            result = minimize(
                neg_ln_like,
                gp.get_parameter_vector(),
                # method = 'Nelder-Mead',
                method = 'L-BFGS-B',
                # method = 'SLSQP',
                # method="BFGS",
                jac=grad_neg_ln_like,
            )
            # print(result)
            gp.set_parameter_vector(result.x)
            hyper_opt.append(np.exp(result.x))
            # print(f"Final log-likelihood: {gp.log_likelihood(y)}")
            # print(result.x, gp.kernel)

            pred, pred_var = gp.predict(y, x_pred, return_var=True)
            r2_pred = r2_score(df_pred[output_label], pred)
            # print(f'log_likelihood:{gp.log_likelihood(y)}')
            # print(f'r2:{r2_pred}')

        except Exception:
            print("opt fail")
        pass

    r2s_opt.append(r2_pred)
    lkhs_opt.append(gp.log_likelihood(y))
    gps.append(gp)


df_coeff = pd.DataFrame(
    np.asarray([r2s, lkhs, r2s_opt, lkhs_opt]).T,
    columns=["r2", "likelihood", "r2_opt", "likelihood_opt"],
)
df_coeff["mae"] = [
    mean_absolute_error(
        df_pred[output_label], gp.predict(y, x_pred, return_var=True)[0]
    )
    for gp in gps
]
df_coeff = pd.concat((df_coeff, pd.DataFrame(trials, columns=["a", "b"])), axis=1)
df_coeff = pd.concat(
    (df_coeff, pd.DataFrame(np.asarray(hyper_opt), columns=["a_opt", "b_opt"])), axis=1
)

df_coeff.sort_values(by="likelihood_opt", ascending=False).head()

# %%
pred, pred_var = gps[df_coeff.idxmax()["likelihood_opt"]].predict(
    y, x_pred, return_var=True
)
plt.plot(df_pred[output_label], pred, "d")
plt.title(f"r2={r2_score(df_pred[output_label], pred)}")

# %%time
pred, pred_var = gps[df_coeff.idxmax()["likelihood"]].predict(
    y, x_pred, return_var=True
)


# %%
mean_absolute_error(pred, df_pred[output_label])
# %%
preds = [gp.predict(y, x_pred, return_var=True)[0] for gp in gps]

# %%
[mean_absolute_error(pred, df_pred[output_label]) for pred in preds]

# %%

# %%
df_coeff
# %%
