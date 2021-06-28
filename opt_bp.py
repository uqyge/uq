# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
import onnxruntime as ort
# from progress.bar import Bar
from scipy.optimize import minimize
# from optimparallel import minimize_parallel
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# import dask
# from dask.distributed import Client

# dask.config.set(scheduler="processes")
# dask.config.set(scheduler="threads")
#%%
# client = Client(n_workers=6, threads_per_worker=1)
# client = Client()
# client

# %%
# model = tf.keras.models.load_model("models/act_abbc_nn.h5")
model_0 = tf.keras.models.load_model("models/act_abbc_nn_010.h5")
model = tf.keras.models.load_model("models/act_abbc_nn_010.h5")

# df_train = pd.read_csv("abbc.csv")
df_train = pd.read_csv("data/0_10.csv")
plt.figure()
# plt.tricontour(tmp[xLabel],tmp[yLabel], z, 15, linewidths=0.1, colors='k')
plt.tricontourf(df_train["23x"], df_train["23y"], df_train["23z"], 15)
plt.xlabel("23x")
plt.ylabel("23y")
plt.colorbar()
plt.savefig("output/test.jpg")

#%%
x_label = ["A1", "A2", "A4"]
X = df_train[x_label]

y_label = ["23x", "23y", "23z"]
Y = df_train[y_label]

input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()

# input_scaler = StandardScaler()
# output_scaler = StandardScaler()

x = input_scaler.fit_transform(X)
y = output_scaler.fit_transform(Y)
#%%
sess = ort.InferenceSession("models/model.onnx")
def ortRun(x_in):
    # x_in = np.ones(3,np.float32).reshape(1,-1)
    x_in = input_scaler.transform(x_in).astype(np.float32)
    results = sess.run([], {"dense_input:0": x_in})
    return output_scaler.inverse_transform(results[0])

#%%
x_test = X.values[:2,:]

a = ortRun(x_test)
#%%

# %%
def convergence_plot(x, y, fname):
    plt.figure()
    plt.plot(x, y, "k")
    plt.plot(x, y, "ko")
    plt.title("Convergence plot")
    plt.xlabel("Number of calls n")
    plt.ylabel("min f(x) after n calls")
    # plt.xscale('log')
    plt.grid()
    plt.savefig(f"output/{fname}.jpg")


def writeOpt(opt_v, fname):
    opt_x = opt_v[["A1", "A23", "A23", "A4"]].values.reshape(1, -1)
    # display(opt_x)
    np.savetxt(f"output/inputACT_abbc_{fname}.csv", opt_x, delimiter=",")
    print(f"input for optimal solution is {opt_x}")
    print(f"optimal solution is {opt_v['pred']}")
    return


def sqp_post(sols, fname):
    data = np.asarray(
        [
            np.hstack(
                (sol.nfev, sol.x.round(3), nn_pred(sol.x.round(3))[0], -sol.fun)
            ).tolist()
            for sol in sols
        ]
    )
    df_opt_sqp = pd.DataFrame(
        data, columns=["nfev", "A1", "A23", "A4", "23x", "23y", "23z", "opt"]
    )

    x = [df_opt_sqp.nfev[: i + 1].sum() for i in range(df_opt_sqp.shape[0])]
    y = [df_opt_sqp.opt[: i + 1].max() for i in range(df_opt_sqp.shape[0])]
    np.savetxt(f"output/sqp_{fname}.csv", np.vstack([x, y]), delimiter=",")
    convergence_plot(x, y, "sqp_" + fname)
    return df_opt_sqp


def nn_pred(x):
    out = model.predict(input_scaler.transform(x.reshape(1, -1)))
    out = output_scaler.inverse_transform(out)
    return out


def obj(x):
    out = nn_pred(x)
    return -out[0, 2]

def obj_ort(x):
    out = ortRun(x.reshape(1,-1))
    return -out[0, 2]


def obj_zy(x):
    out = nn_pred(x)
    return -abs(out[0, 2]) / (abs(out[0, 1]) + abs(out[0, 2]))


def obj_zy_ort(x):
    out = ortRun(x.reshape(1,-1))
    return -abs(out[0, 2]) / (abs(out[0, 1]) + abs(out[0, 2]))


def constraint(x):
    out = nn_pred(x)
    return 1 - abs(out[0, 1])


cons = {"type": "ineq", "fun": lambda x: 1 - abs(nn_pred(x)[0, 1])}
# cons = ({'type': 'eq', 'fun': lambda x:abs(nn_pred(x)[0,1])})


def callback(x):
    # print(f"hello:{x}")
    # print(f"obj_zy:{obj_zy(x)}")
    # print(f"obj:{obj(x)}")
    # fobj = obj(x)
    # history.append(fobj)
    # x_evl.append(x)
    return

#%%
# obj_zy_ort([0.1,0.1,0.1])
#%%

# from multiprocessing.dummy import Pool
import multiprocessing as mp
import threading
mp.set_start_method('spawn',force=True)
lock = threading.Lock()
#%%
%%time
n_trials = 20
trials = np.random.uniform(0.01, 0.1, 3 * n_trials).reshape(n_trials, -1)
# print(trials)

old_w = model.get_weights()
sols = []
def f(x_ini):
    # model = tf.keras.models.load_model("models/act_abbc_nn_010.h5")

    # model = tf.keras.models.clone_model(model_0)
    # model.set_weights(old_w)
    # lock.acquire()
    def obj_pl(x):
        # import tensorflow as tf

        # model = tf.keras.models.load_model("models/act_abbc_nn_010.h5")
        # out = model.predict(input_scaler.transform(x.reshape(1, -1)))
        # out = output_scaler.inverse_transform(out)
        # print(x)
        out = ortRun(x.reshape(1,-1))
        return -out[0, 2]
        # return x.reshape(1,-1)[0,2]
    # lock.release()

    # x0 = np.copy(x_ini)
    # lock.acquire()
    sol = minimize(
        obj_ort,
        x_ini,
        bounds=(
            (0.01, 0.1),
            (0.01, 0.1),
            (0.01, 0.1),
        ),
        # constraints=cons,
        method="SLSQP",
        # method="L-BFGS-B",
        options={"disp": False, "maxiter": 200},
    )
    # lock.release()

    return sol

for i in trials:
    sol = f(i)
    sols.append(sol)

[i.fun for i in sols]
#%%
f(trials[0])

#%%
with mp.Pool(1) as p:
    # mp.set_start_method('spawn',force=True)
    sols = p.map(f, trials)
    # print(a)

[i.fun for i in sols]
# %% case 1 zmax
%%time
sols = []
n_trials = 20
trials = np.random.uniform(0.01, 0.1, 3 * n_trials).reshape(n_trials, -1)
# with dask.config.set(scheduler="processes"):
for x0 in trials:
    # print(x0)
    sol = minimize(
        obj,
        x0,
        bounds=(
            (0.01, 0.1),
            (0.01, 0.1),
            (0.01, 0.1),
        ),
        # constraints=cons,
        method="SLSQP",
        # method = 'L-BFGS-B',
        options={"disp": False, "maxiter": 200},
        # callback=callback,
    )
    sols.append(sol)

# sols = dask.compute(*sols)

[i.fun for i in sols]
#%%
df_opt_sqp = sqp_post(sols, "zmax")
df_opt_sqp["pred"] = df_opt_sqp["23z"]
opt_v = df_opt_sqp.iloc[df_opt_sqp["pred"].idxmax()]
opt_v.to_csv("output/sqp_zmax_nn.csv")
writeOpt(opt_v, "sqp_zmax")


# %% case 2 z/y ratio
sols = []
x_evl = []
# bar = Bar("processing", max=trials)
for i in range(20):
    # bar.next()
    x0 = np.random.uniform(0.0, 0.1, 3).reshape(1, -1)
    sol = minimize(
        # obj_zy,
        obj_zy_ort,
        x0,
        bounds=(
            (0.0, 0.1),
            (0.0, 0.1),
            (0.0, 0.1),
        ),
        # constraints=cons,
        method="SLSQP",
        # method = 'L-BFGS-B',
        options={"disp": False, "maxiter": 200},
        callback=callback,
    )
    sols.append(sol)
# bar.finish()

#%%
df_opt_sqp = sqp_post(sols, "zyratio")
df_opt_sqp["pred"] = abs(df_opt_sqp["23z"]) / (
    abs(df_opt_sqp["23y"]) + abs(df_opt_sqp["23z"])
)
opt_v = df_opt_sqp.iloc[df_opt_sqp["pred"].idxmax()]
opt_v.to_csv("output/sqp_zyratio_nn.csv")
writeOpt(opt_v, "sqp_zyratio")
opt_v
# df_opt_sqp.head()

#%% shuffle SQP convergence plot for postprocessing
fname = "zmax"
df_shuffle = df_opt_sqp.sample(frac=1)
x = [df_shuffle.nfev[: i + 1].sum() for i in range(df_opt_sqp.shape[0])]
y = [df_shuffle.opt[: i + 1].max() for i in range(df_opt_sqp.shape[0])]
np.savetxt(f"output/sqp_{fname}.csv", np.vstack([x, y]), delimiter=",")
convergence_plot(x, y, "sqp_" + fname)


#%%
#######################################
###### Bayesian optimization ##########
#######################################

#%%
from skopt import gp_minimize
from skopt.plots import plot_convergence


def nn_pred_list(x):
    out = model.predict(input_scaler.transform(np.asarray(x).reshape(1, -1)))
    out = output_scaler.inverse_transform(out)
    return out


def obj_gp(x):
    return -nn_pred_list(x)[0, 2]


def obj_gp_ort(x):
    # return -nn_pred_list(x)[0, 2]
    return -ortRun(np.asarray(x).reshape(1,-1))[0,2]


def obj_gp_zy(x):
    # return -abs(nn_pred_list(x)[0, 2]) / (abs(nn_pred_list(x)[0, 1]) + 100)
    return -abs(nn_pred_list(x)[0, 2]) / (
        abs(nn_pred_list(x)[0, 1]) + abs(nn_pred_list(x)[0, 2])
    )


def gp_post(res, fname):
    x_round_i = np.asarray(res.x_iters).round(3)
    opt_org_i = -1 * res.func_vals.reshape(-1, 1)
    prd_round_i = np.asarray([nn_pred_list(x).tolist()[0] for x in x_round_i])

    df_opt_gp = pd.DataFrame(
        np.concatenate(
            [
                x_round_i,
                prd_round_i,
                opt_org_i,
            ],
            axis=1,
        ),
        columns=[
            "A1",
            "A23",
            "A4",
            "23x",
            "23y",
            "23z",
            "opt",
        ],
    )
    x = df_opt_gp.index.values + 1
    y = [df_opt_gp.opt[: i + 1].max() for i in range(df_opt_gp.shape[0])]
    np.savetxt(f"output/gp_{fname}.csv", np.vstack([x, y]), delimiter=",")

    convergence_plot(x, y, "gp_" + fname)

    return df_opt_gp

#%%
x0 = [0,0,0]

print(obj_gp(x0))
print(obj_gp_ort(x0))
# %% case 1 zmax
%%time
res = gp_minimize(
    obj_gp_ort, [(0.0, 0.1), (0.0, 0.1), (0.0, 0.1)], acq_func="EI", n_calls=50
)
# %% case 1 zmax
%%time
res = gp_minimize(
    obj_gp, [(0.0, 0.1), (0.0, 0.1), (0.0, 0.1)], acq_func="EI", n_calls=50
)
#%%
df_opt_gp = gp_post(res, "zmax")
df_opt_gp["pred"] = df_opt_gp["23z"]
opt_v = df_opt_gp.iloc[df_opt_gp["pred"].idxmax()]
opt_v.to_csv("output/gp_zmax_nn.csv")
writeOpt(opt_v, "gp_zmax")

#%%
df_opt_gp


#%% case 2 z/y ratio
res = gp_minimize(
    obj_gp_zy,
    [(0.01, 0.1), (0.01, 0.1), (0.01, 0.1)],
    acq_func="EI",
    n_calls=100,
)

#%%
df_opt_gp = gp_post(res, "zyratio")
df_opt_gp["pred"] = abs(df_opt_gp["23z"]) / (
    abs(df_opt_gp["23y"]) + abs(df_opt_gp["23z"])
)
opt_v = df_opt_gp.iloc[df_opt_gp["pred"].idxmax()]
opt_v.to_csv("output/gp_zyratio_nn.csv")
writeOpt(opt_v, "gp_zyratio")

# %%
plt.plot(df_opt_gp['opt'],'d')
# %%
