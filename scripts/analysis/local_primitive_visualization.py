"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffused_fields_robotics.
Licensed under the MIT License. See LICENSE file in the project root.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from diffused_fields_robotics.core.config import get_plots_dir

# Input data
data = (
    np.array(
        [
            [1.13670868e-04, -5.85979276e-03, -3.82751979e-03],
            [-1.29991103e-03, -5.75003860e-03, -3.77455792e-03],
            [-1.17822879e-03, -5.69989898e-03, -3.88882098e-03],
            [-9.25811193e-05, -5.67399066e-03, -4.09844589e-03],
            [-5.00207250e-04, -5.70335111e-03, -4.02760212e-03],
            [3.56242833e-04, -6.02300159e-03, -3.54916087e-03],
            [2.65556823e-04, -5.70568413e-03, -4.04656004e-03],
            [7.60084767e-04, -5.63980291e-03, -4.07613718e-03],
            [6.26077501e-04, 4.07702126e-03, -5.65561001e-03],
            [1.77067381e-04, 2.92113577e-03, -6.35890029e-03],
            [6.54538942e-05, 2.10394075e-03, -6.67601296e-03],
            [-2.21134205e-05, 1.14489199e-03, -6.90570296e-03],
            [-2.86210046e-05, 7.50367320e-04, -6.95960701e-03],
            [6.45160875e-05, 9.98122029e-04, -6.92817365e-03],
            [1.71786450e-04, 1.77857484e-03, -6.76809877e-03],
            [2.32516124e-04, 2.32854469e-03, -6.59725821e-03],
            [6.98334380e-03, 3.91438013e-04, 2.82286397e-04],
            [6.98923180e-03, -3.69875224e-04, 1.17606105e-04],
            [6.89163462e-03, -1.22094640e-03, -1.21087413e-04],
            [6.55336478e-03, -2.40193973e-03, -5.33006176e-04],
            [5.74986984e-03, -3.85481476e-03, -1.03894177e-03],
            [4.96814945e-03, -4.77890132e-03, -1.21638527e-03],
            [4.33906474e-03, -5.34121546e-03, -1.28216011e-03],
            [3.71414930e-03, -5.82642581e-03, -1.12154238e-03],
        ]
    )
    * 1.5e2
)
data_tmp = np.copy(data)
data[:, 2] = -data_tmp[:, 1]
data[:, 1] = data_tmp[:, 0]
data[:, 0] = -data_tmp[:, 2]

# Base signal setup
base_x = [0, 0, 1]
base_y = [0, 0, 0]
base_z = [-1, 1, 0]

base_x = [1, 0, -1, 0, 0]
base_y = [0, 0, 0, 1, 0]
base_z = [0, -1, 0, 0, 1]
repeat = 8
n_cycles = 1

# Generate latched and repeated signals
x = np.tile(np.repeat(base_x, repeat), n_cycles)
y = np.tile(np.repeat(base_y, repeat), n_cycles)
z = np.tile(np.repeat(base_z, repeat), n_cycles)
n_steps = len(x)

signals = [x, y, z]
base_signals = [base_x, base_y, base_z]
axis_labels = ["X", "Y", "Z"]
colors = ["red", "green", "blue"]

# Create tight subplots with shared x-axis
fig, axes = plt.subplots(3, 1, figsize=(4, 4), sharex=True, gridspec_kw={"hspace": 0.0})

sweep_lines = []

for i in range(3):
    ax = axes[i]

    # Step signal
    ax.step(
        np.arange(n_steps + 1),
        np.append(signals[i], signals[i][-1]),
        where="post",
        color=colors[i],
        linewidth=5,
        alpha=1.0,
        label="Local frame",
    )

    # # Actual data as dashed line
    # ax.plot(
    #     np.arange(data.shape[0]),
    #     data[:, i],
    #     linestyle="--",
    #     linewidth=2,
    #     color=colors[i],
    #     alpha=0.6,
    #     label="World frame",
    # )

    # Zero line
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)

    # Axis settings
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")

    # Axis label
    ax.text(
        -0.01,
        0.5,
        axis_labels[i],
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    # Animated sweep line
    (line,) = ax.plot([0, 0], [-1.2, 1.2], "k--", linewidth=3)
    sweep_lines.append(line)

    # Vertical lines at changes
    base = base_signals[i]
    change_indices = [j for j in range(1, len(base)) if base[j] != base[j - 1]]
    # for idx in change_indices:
    #     xpos = idx * repeat
    #     ax.axvline(x=xpos, linestyle="--", color="gray", linewidth=2, alpha=1.0)

    # Legend
    # ax.legend(loc="upper right", fontsize=8)


# Animation function
def update(frame):
    for line in sweep_lines:
        # continue
        line.set_xdata([frame, frame])
    return sweep_lines


# Animate
anim = FuncAnimation(
    fig, update, frames=np.arange(n_steps + 1), interval=150, blit=True
)

plt.subplots_adjust(left=0.05, right=1, top=1, bottom=0, hspace=0)

# Save static figure
plots_dir = get_plots_dir()
fig.savefig(
    plots_dir / "minimal_signal_static_with_data.pdf", dpi=300, bbox_inches="tight"
)

# Show animation
plt.show()

# Optional: Save animation
anim.save("minimal_signal_sweep_with_data.mp4", writer="ffmpeg", fps=5)
