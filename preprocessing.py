# %%
import pandas as pd

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
df = pd.read_parquet("data/最终.parquet")
