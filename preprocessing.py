# %%
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from SALib.sample import saltelli


# %%
def convert_parquet(f_name):
    """Convert xlsx & csv to parquet format"""

    c_name, surfix = f_name.split("/")[-1].split(".")
    if surfix == "xlsx":
        df = pd.read_excel(f_name)
    elif surfix == "csv":
        df = pd.read_csv(f_name)
    else:
        print("not supported")
        return

    df.set_axis([str(col) for col in df.columns], axis=1, inplace=True)
    df.to_parquet(f"data/{c_name}.parquet")

    assert df.shape == pd.read_parquet(f"data/{c_name}.parquet").shape


# %%
# convert_parquet("data/0_10.csv")
#%%
df = pd.read_parquet("data/最终.parquet")
df_2 = pd.read_parquet("data/0_10.parquet")

# %%
px.scatter_3d(df_2.sample(3_000), x="A1", y="A2", z="A4", color="23z")
# %%
px.scatter_3d(df.sample(3_000), x="A1", y="A2", z="A4", color="23z")

#%%
px.scatter_3d(df.sample(3_000), x="A1", y="A2", z="A4", color="31")
#%%
px.scatter_3d(df_2.sample(3_000), x="A1", y="A2", z="A4", color="31")
# %%
# actuation = {"num_vars": 3, "names": ["A1", "A2", "A4"], "bounds": [[0, 0.3]] * 3}
# p = saltelli.sample(actuation, 10_000)
# df_sobol = pd.DataFrame(p, columns=actuation["names"])
# px.scatter_3d(df_sobol.sample(3_000), x="A1", y="A2", z="A4")

# %%
px.scatter_3d(df[abs(df["ave18_23y"]) < 1], x="A1", y="A2", z="A4", color="ave18_23y")
# %%
px.scatter_3d(df[abs(df["ave18_23y"]) < 1], x="A1", y="A2", z="A4", color="ave18_23z")
# %%
px.scatter_3d(df[abs(df["ave18_23y"]) < 1], x="A1", y="A2", z="A4", color="31")
# %%
px.scatter_3d(df_2[abs(df_2["23y"]) < 1], x="A1", y="A2", z="A4", color="23y")
# %%
px.scatter_3d(df_2[abs(df_2["23y"]) < 1], x="A1", y="A2", z="A4", color="23z")
# %%
px.scatter_3d(df_2[abs(df_2["23y"]) < 1], x="A1", y="A2", z="A4", color="31")

# %%
cond = (df.A1 < 0.1) & (df.A2 < 0.1) & (df.A4 < 0.1)
px.scatter_3d(df[cond], x="A1", y="A2", z="A4", color="23z")


# %%
df_tmp = df[cond]
px.scatter_3d(df_tmp[abs(df_tmp["ave18_23y"]) < 1], x="A1", y="A2", z="A4", color="ave18_23y")



# %%
