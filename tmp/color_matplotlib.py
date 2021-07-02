# %%
import matplotlib.pyplot as plt

# line cyclers adapted to colourblind people
from cycler import cycler
line_cycler   = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-."]))
marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
                 cycler(marker=["4", "2", "3", "1", "+", "x", "."]))

# matplotlib's standard cycler
standard_cycler = cycler("color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])

plt.rc("axes", prop_cycle=line_cycler)

# plt.rc("text", usetex=True)
# plt.rc("text.latex", preamble=r"\usepackage{newpxtext}\usepackage{newpxmath}\usepackage{commath}\usepackage{mathtools}")
# plt.rc("font", family="serif", size=18.)
plt.rc("savefig", dpi=200)
plt.rc("legend", loc="best", fontsize="medium", fancybox=True, framealpha=0.5)
plt.rc("lines", linewidth=2.5, markersize=10, markeredgewidth=2.5)

#%%
import os
directory = os.path.dirname(os.path.abspath(__file__))


# plots
plt.close("all")

import numpy as np
x = np.linspace(0., 1., 200)

plt.rc("axes", prop_cycle=line_cycler)
plt.figure()
plt.plot(x,   np.sin(np.pi*x), label="$\sin(\pi x)$")
plt.plot(x,   np.cos(np.pi*x), label="$\cos(\pi x)$")
plt.plot(x, 2*np.sin(np.pi*x), label="$2 \sin(\pi x)$")
plt.plot(x, 2*np.cos(np.pi*x), label="$2 \cos(\pi x)$")
plt.legend()
plt.xlabel("$x$")
plt.xlim(x[0], x[-1])
plt.ylabel("Function Values")
# plt.savefig(os.path.join(directory, "line_plot_python.png"), bbox_inches="tight")

plt.rc("axes", prop_cycle=standard_cycler)
plt.figure()
plt.plot(x,   np.sin(np.pi*x), label="$\sin(\pi x)$")
plt.plot(x,   np.cos(np.pi*x), label="$\cos(\pi x)$")
plt.plot(x, 2*np.sin(np.pi*x), label="$2 \sin(\pi x)$")
plt.plot(x, 2*np.cos(np.pi*x), label="$2 \cos(\pi x)$")
plt.legend()
plt.xlabel("$x$")
plt.xlim(x[0], x[-1])
plt.ylabel("Function Values")
plt.savefig(os.path.join(directory, "line_plot_python_standard.png"), bbox_inches="tight")

#%%
import numpy as np
x = np.linspace(0., 1., 20)

plt.rc("axes", prop_cycle=marker_cycler)
plt.figure()
plt.plot(x,   np.sin(np.pi*x), label="$\sin(\pi x)$")
plt.plot(x,   np.cos(np.pi*x), label="$\cos(\pi x)$")
plt.plot(x, 2*np.sin(np.pi*x), label="$2 \sin(\pi x)$")
plt.plot(x, 2*np.cos(np.pi*x), label="$2 \cos(\pi x)$")
plt.legend()
plt.xlabel("$x$")
plt.xlim(x[0], x[-1])
plt.ylabel("Function Values")
plt.savefig(os.path.join(directory, "marker_plot_python.png"), bbox_inches="tight")

plt.rc("axes", prop_cycle=standard_cycler)
plt.rc("lines", linewidth=1.5, markersize=6, markeredgewidth=1)
plt.figure()
plt.scatter(x,   np.sin(np.pi*x), label="$\sin(\pi x)$")
plt.scatter(x,   np.cos(np.pi*x), label="$\cos(\pi x)$")
plt.scatter(x, 2*np.sin(np.pi*x), label="$2 \sin(\pi x)$")
plt.scatter(x, 2*np.cos(np.pi*x), label="$2 \cos(\pi x)$")
plt.legend()
plt.xlabel("$x$")
plt.xlim(x[0], x[-1])
plt.ylabel("Function Values")
plt.savefig(os.path.join(directory, "marker_plot_python_standard.png"), bbox_inches="tight")
# %%
