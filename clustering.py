#%%
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# %%
# df = pd.read_excel("./data/9_10_150_100000.xlsx")
df = pd.read_csv("./data/9_10.csv")
# %%
px.scatter_3d(df.sample(1_000), x="23x", y="23y", z="23z")

#%%
((df["A14"] - df["A23"]) > 0).sum()

#%%
df.head()
# %%
import hdbscan

# %%
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
clusterer.fit(df[["23x", "23y", "23z"]])
df["class"] = clusterer.labels_
df["class"] = df["class"].astype("category")
tmp = df.groupby("class")
tmp.count()
# %%
df.head()
# %%
sample_size = min(30_000, df.shape[0])

fig = px.scatter_3d(df.sample(sample_size), x="23x", y="23y", z="23z", color="class")

fig.update_traces(
    marker=dict(size=2, line=dict(width=1, color="DarkSlateGrey")),
    selector=dict(mode="markers"),
)
fig.show()
# %%
df.to_csv("./data/9_10.csv")
# %%
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np

# Define the model inputs
problem = {
    "num_vars": 3,
    "names": ["x1", "x2", "x3"],
    "bounds": [
        [-3.14159265359, 3.14159265359],
        [-3.14159265359, 3.14159265359],
        [-3.14159265359, 3.14159265359],
    ],
}

# Generate samples
param_values = saltelli.sample(problem, 1000)

# Run model (example)
Y = Ishigami.evaluate(param_values)

# Perform analysis
Si = sobol.analyze(problem, Y, print_to_console=True)

# Print the first-order sensitivity indices
print(Si["S1"])
# %%
Y
# %%
problem
# %%
param_values.shape
# %%
prob = {"num_vars": 2, "names": ["x1", "x2"], "bounds": [[0.09, 0.1], [0.09, 0.1]]}

Y_sob = df["23z"].values
# %%
sobol.analyze(prob, Y_sob, print_to_console=True)
# %%
df_sol=df.sample(12_000)
Y_sob = df_sol["23z"].values

input_names = ["A14", "A23", "Tup", "Tmid", "Tdown"]
input_bounds = [[0.09, 0.1], [0.09, 0.1], [0, 150], [0, 150], [0, 150]]
prob = {"num_vars": 5, "names": input_names, "bounds": input_bounds}
param_values = saltelli.sample(prob, 1000)

sobol.analyze(prob, Y_sob, print_to_console=True)
# %%
df['23z'].sample(8_000).plot.hist()





