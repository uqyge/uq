#%%
import pandas as pd
import plotly.express as px

# %%
# df = pd.read_excel("data/全筛选.xlsx")
# str_cols = [str(col) for col in df.columns]
# df.set_axis(str_cols, axis=1, inplace=True)
# df.to_parquet("quanshaixuan.parquet")

# %%
# df = pd.read_parquet("no_g_full.parquet")
# df = pd.read_parquet("./no_g_handpick.parquet")
df = pd.read_parquet("./quanshaixuan.parquet")
#%%
df.describe()
#%%
cable_stresses = [str(i + 1) for i in range(32)]
stress_idxmax = df[cable_stresses].idxmax(axis=1)
display(stress_idxmax.value_counts())

for idx in set(stress_idxmax):
    print(idx, df[stress_idxmax == idx][idx].max())

# %%
df_plot = df.sample(10_000)
cond_stress = df_plot[cable_stresses].max(axis=1) < 500

fig = px.scatter_3d(
    df_plot[cond_stress],
    x="23x",
    y="23y",
    z="23z",
    color=df_plot[cond_stress][cable_stresses].max(axis=1),
)
fig.update_traces(
    marker=dict(size=2, line=dict(width=1, color="DarkSlateGrey")),
    selector=dict(mode="markers"),
)
fig.show()
# %%

# %%
df.ijk.plot.pie()
# %%
df[["1", "2", "3"]].max(axis=1)
# %%

# %%
df.columns
# %%

#%%
df_plot[stress].max(axis=1)
# %%
df_plot[cond_stress]
# %%
df_plot[cond & cond_stress]
# %%

#%%
import multiprocessing as mp
import pickle

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from pyDOE import lhs
from scipy.optimize import minimize
from skopt import gp_minimize
import sys

# if not sys.warnoptions:
#     import warnings
#     warnings.simplefilter("ignore")

#%%
# df_train = pd.read_csv("data/0_10.csv")
# cond = (df_train[['A1','A2']].min(axis=1)>0.01)
# df_plt = df_train[cond].sample(10_000)
# # df_plt = df_train.sample(10_000)

# plt.figure()
# plt.tricontourf(df_plt["23x"], df_plt["23y"], df_plt["23z"], 15)
# plt.xlabel("23x")
# plt.ylabel("23y")
# plt.colorbar()
# plt.savefig("output/test.jpg")

#%%
with open('models/minMax_quanshaixuan.pkl','rb') as f:
    input_scaler, output_scaler = pickle.load(f)

sess = ort.InferenceSession("models/model_quanshaixuan.onnx")
def ortRun(x_in):
    x_in = input_scaler.transform(x_in).astype(np.float32)
    results = sess.run([], {"dense_input:0": x_in})
    return output_scaler.inverse_transform(results[0])

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

def hist_plot(fname):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    df_opt_gp['pred'].plot.hist(bins=20,grid=True)
    plt.title('gp')
    plt.subplot(1,2,2)
    df_opt_sqp.sample(df_opt_gp.shape[0])['pred'].plot.hist(bins=20,grid=True)
    plt.title('sqp')
    plt.savefig(f'output/{fname}.jpg')


def writeOpt(opt_v, fname):
    opt_x = opt_v[["A1", "A23", "A23", "A4"]].values.reshape(1, -1)
    # display(opt_x)
    np.savetxt(f"output/inputACT_abbc_{fname}.csv", opt_x, delimiter=",")
    print(f"input for optimal solution is {opt_x}")
    print(f"optimal solution is {opt_v['pred']}")
    return


def gp_post(res, fname):
    x_round_i = np.asarray(res.x_iters).round(3)
    opt_org_i = -1 * res.func_vals.reshape(-1, 1)
    prd_round_i = np.asarray([ortRun(x.reshape(1,-1)).tolist()[0] for x in x_round_i])

    df_opt_gp = pd.DataFrame(
        np.concatenate(
            [
                np.ones(len(x_round_i)).reshape(-1,1),
                x_round_i,
                prd_round_i,
                opt_org_i,
            ],
            axis=1,
        ),
        columns=[
            "nfev",
            "A1",
            "A23",
            "A4",
            "23x",
            "23y",
            "23z",
            "opt",
        ],
    )
    df_opt_gp['method']='gp'
    x = df_opt_gp.index.values + 1
    y = [df_opt_gp.opt[: i + 1].max() for i in range(df_opt_gp.shape[0])]
    np.savetxt(f"output/gp_{fname}.csv", np.vstack([x, y]), delimiter=",")

    convergence_plot(x, y, "gp_" + fname)

    return df_opt_gp


def sqp_post(sols, fname):
    data = np.asarray(
        [
            np.hstack(
                (sol.nfev, sol.x.round(3), ortRun(sol.x.round(3).reshape(1,-1))[0], -sol.fun)
            ).tolist()
            for sol in sols
        ]
    )
    df_opt_sqp = pd.DataFrame(
        data, columns=["nfev", "A1", "A23", "A4", "23x", "23y", "23z", "opt"]
    )
    df_opt_sqp['method']='sqp'
    x = [df_opt_sqp.nfev[: i + 1].sum() for i in range(df_opt_sqp.shape[0])]
    y = [df_opt_sqp.opt[: i + 1].max() for i in range(df_opt_sqp.shape[0])]
    np.savetxt(f"output/sqp_{fname}.csv", np.vstack([x, y]), delimiter=",")
    convergence_plot(x, y, "sqp_" + fname)
    return df_opt_sqp

def obj_z_ort(x):
    x=np.asarray(x)
    return -ortRun(x.reshape(1,-1))[0,2]

def obj_y_ort(x):
    x=np.asarray(x)
    out = ortRun(x.reshape(1,-1))
    return -abs(out[0, 1])

def obj_zy_ort(x):
    x=np.asarray(x)
    out = ortRun(x.reshape(1,-1))
    return -abs(out[0, 2]) / (abs(out[0, 1]) + abs(out[0, 2]))

# def constraint(x):
#     out = nn_pred(x)
#     return 1 - abs(out[0, 1])
# cons = {"type": "ineq", "fun": lambda x: 1 - abs(nn_pred(x)[0, 1])}
# cons = ({'type': 'eq', 'fun': lambda x:abs(nn_pred(x)[0,0.1])})

def sqp(func_obj,x_ini):
    sol = minimize(
        func_obj,
        x_ini,
        bounds=(
            (0.01, 0.3),
            (0.01, 0.3),
            (0.01, 0.3),
        ),
        # constraints={"type": "ineq", "fun": lambda x: 100 - ortRun(x.reshape(1,-1))[0,0]},
        method="SLSQP",
        # method="L-BFGS-B",
        options={"disp": False, "maxiter": 200},
    )

    return sol



def callback(x):
    # print(f"hello:{x}")
    # fobj = obj(x)
    # history.append(fobj)
    return

#%% case 1 z max
%%time

lb = np.array([0.01,0.01,0.01])
ub = np.array([0.2,0.2,0.2])

n_trials = 200
trials = lb+(ub-lb)*lhs(3,n_trials).round(3)

def test(x_ini):
    return sqp(obj_z_ort,x_ini)

sols=[]
with mp.Pool() as p:
    sols = p.map(test,trials)

df_opt_sqp = sqp_post(sols, "zmax")
df_opt_sqp["pred"] = df_opt_sqp["23z"]
opt_v = df_opt_sqp.iloc[df_opt_sqp["pred"].idxmax()]
opt_v.to_csv("output/sqp_zmax_nn.csv")
writeOpt(opt_v, "sqp_zmax")
opt_v

# %%
lb = np.array([0.01,0.01,0.01])
ub = np.array([0.3,0.3,0.3])

n_trials = 1000
trials = lb+(ub-lb)*lhs(3,n_trials).round(3)


def test(x_ini):
    return sqp(obj_zy_ort,x_ini)

sols=[]
with mp.Pool() as p:
    sols=p.map(test,trials)

df_opt_sqp = sqp_post(sols, "zyratio")
df_opt_sqp["pred"] = abs(df_opt_sqp["23z"]) / (
    abs(df_opt_sqp["23y"]) + abs(df_opt_sqp["23z"])
)
opt_v = df_opt_sqp.iloc[df_opt_sqp["pred"].idxmax()]
opt_v.to_csv("output/sqp_zyratio_nn.csv")
writeOpt(opt_v, "sqp_zyratio")
# opt_v
df_opt_sqp.iloc[df_opt_sqp.sort_values(by=['pred'],ascending=False).head(10)['23z'].idxmax()]

# %%
ortRun(np.asarray([0.2,0.2,0.02]).reshape(1,-1))
# %%
ortRun(np.asarray([0.0,0.0,0.02]).reshape(1,-1))
# %%
