# %%
import numpy as np
from pyvista import examples

# %%
# Load a simple example mesh
dataset = examples.load_uniform()
dataset.set_active_scalars("Spatial Cell Data")
# %%
# Compute volumes and areas
sized = dataset.compute_cell_sizes()

# Grab volumes for all cells in the mesh
cell_volumes = sized.cell_arrays["Volume"]
# %%
volume = dataset.volume
# %%
threshed = dataset.threshold_percent([0.15, 0.50], invert=True)
threshed.plot(show_grid=True, cpos=[-2, 5, 3])

# %%
