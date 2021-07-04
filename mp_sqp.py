# %%
import multiprocessing as mp
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from pyDOE import lhs
from scipy.optimize import minimize

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


#%%
f_model = "models/最终.onnx"
sess = ort.InferenceSession(f_model)
f_scaler = "models/最终_minMax.pkl"
with open(f_scaler, "rb") as f:
    input_scaler, output_scaler = pickle.load(f)


def ortRun(x_in):
    # x_in = input_scaler.transform(x_in)
    results = sess.run([], {"rescaling_input": x_in.astype(np.float32)})
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
                (
                    sol.success,
                    sol.nfev,
                    sol.x.round(3),
                    ortRun(sol.x.round(3).reshape(1, -1))[0],
                    -sol.fun,
                )
            ).tolist()
            for sol in sols
        ]
    )
    df_opt_sqp = pd.DataFrame(
        data, columns=["success", "nfev", "A1", "A23", "A4", "31", "23y", "23z", "opt"]
    )
    df_opt_sqp["method"] = "sqp"
    x = [
        df_opt_sqp[df_opt_sqp.success == True].nfev[: i + 1].sum()
        for i in range(df_opt_sqp[df_opt_sqp.success == True].shape[0])
    ]
    y = [
        df_opt_sqp[df_opt_sqp.success == True].opt[: i + 1].max()
        for i in range(df_opt_sqp[df_opt_sqp.success == True].shape[0])
    ]
    np.savetxt(f"output/sqp_{fname}.csv", np.vstack([x, y]), delimiter=",")
    convergence_plot(x, y, "sqp_" + fname)
    return df_opt_sqp


def obj_z_ort(x):
    x = np.asarray(x).reshape(1, -1)
    return -ortRun(x)[0, 2]


def obj_y_ort(x):
    x = np.asarray(x).reshape(1, -1)
    out = ortRun(x)
    return -abs(out[0, 1])


def obj_zy_ort(x):
    x = np.asarray(x).reshape(1, -1)
    out = ortRun(x)
    return -abs(out[0, 2]) / (abs(out[0, 1]) + abs(out[0, 2]))


def sqp(func_obj, x_ini):
    cons = [
        {
            "type": "ineq",
            "fun": lambda x: 1 - abs(ortRun(np.asarray(x).reshape(1, -1))[0, 1]),
        },
        {
            "type": "ineq",
            "fun": lambda x: ortRun(np.asarray(x).reshape(1, -1))[0, 0] - 0,
        },
    ]

    sol = minimize(
        func_obj,
        x_ini,
        bounds=((0.01, 0.3),) * 3,
        # constraints=cons,
        method="SLSQP",
        # method="L-BFGS-B",
        options={"disp": False, "maxiter": 200},
    )

    return sol


#%%
lb = np.array([0.01] * 3)
ub = np.array([0.3] * 3)

n_trials = 1000
trials = lb + (ub - lb) * lhs(3, n_trials).round(3)


def test(x_ini):
    return sqp(obj_z_ort, x_ini)


sols = []
with mp.Pool() as p:
    sols = p.map(test, trials)

df_opt_sqp = sqp_post(sols, "zyratio")
df_opt_sqp["pred"] = abs(df_opt_sqp["23z"]) / (
    abs(df_opt_sqp["23y"]) + abs(df_opt_sqp["23z"])
)

opt_v = df_opt_sqp.iloc[
    df_opt_sqp[df_opt_sqp.success == True]
    .sort_values(by=["pred"], ascending=False)
    .head(10)["23z"]
    .idxmax()
]
print(
    f'optimized actuations = {opt_v[["A1", "A23", "A4"]].values.astype(np.float32).round(3)}'
)
display(opt_v)
opt_v.to_csv("output/sqp_zyratio_nn.csv")
writeOpt(opt_v, "sqp_zyratio")


# %%
df_opt_sqp.sort_values("23z", ascending=False)
# %%
ortRun(np.asarray([0.3, 0.3, 0.173]).reshape(1, -1).astype(np.float32))
# %%

