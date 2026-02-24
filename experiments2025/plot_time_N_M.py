import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plotly.colors import qualitative


palette = qualitative.Plotly
palette = [palette[i] for i in [1, 0, 2, 4, 3,5]]
faded_palette = [color + "66" for color in palette]

df = pd.read_csv("experiments_ILP.csv")

M = [50,100,150,250]
N = [5,10,15,20,30,50]

plt.figure(figsize=(6, 3))
for i, m in enumerate(M):
    runtime_values = []
    averages = []
    for n in N:
        filtered_df = df[(df["M"] == m) & (df["model"] == "ILP")& (df["N"] == n)]
        runtime_values.append(filtered_df["CPU time"].values)
        averages.append(np.mean(filtered_df["CPU time"].values))
        # runtime_values.append(filtered_df["seats"].values)
        # averages.append(np.mean(filtered_df["seats"].values))
    print(averages)
    print(len(averages))

    plt.plot(N, averages, color=palette[i])

    # plt.plot(num_students_values, averages, color=palette[i], marker="o")
    flierprops = dict(markeredgecolor=palette[i])
    box = plt.boxplot(
        runtime_values,
        positions=N,
        patch_artist=True,
        widths=1,
        flierprops=flierprops,
    )
    for whisker, cap in zip(
        box["whiskers"],
        box["caps"],
    ):
        whisker.set_color(palette[i])  # Set whisker color
        whisker.set_linewidth(1.5)  # Make whiskers thicker
        cap.set_color(palette[i])  # Set cap color
        cap.set_linewidth(1.5)
    plt.xticks([])
    # Customize the boxes
    for patch in box["boxes"]:
        patch.set_facecolor(faded_palette[i])
        patch.set_edgecolor(palette[i])  # Set vibrant edge color (no alpha)
        patch.set_linewidth(1.5)  # Edge thickness
        # patch.set_alpha(0.4)
        # Customize the median line
    for median in box["medians"]:
        median.set_color(palette[i])  # Set median line color
        median.set_linewidth(1.5)  # Make the line thicker
    for flier in box["fliers"]:
        flier.set_color(palette[i])

M = [400]
N = [5,10,15,20,30]

for m in M:
    runtime_values = []
    averages = []
    for n in N:
        filtered_df = df[(df["M"] == m) & (df["model"] == "ILP")& (df["N"] == n)]
        runtime_values.append(filtered_df["CPU time"].values)
        averages.append(np.mean(filtered_df["CPU time"].values))
        # runtime_values.append(filtered_df["seats"].values)
        # averages.append(np.mean(filtered_df["seats"].values))
    print(averages)
    print(len(averages))

    plt.plot(N, averages, color=palette[i+1])

    # plt.plot(num_students_values, averages, color=palette[i], marker="o")
    flierprops = dict(markeredgecolor=palette[i+1])
    box = plt.boxplot(
        runtime_values,
        positions=N,
        patch_artist=True,
        widths=1,
        flierprops=flierprops,
    )
    for whisker, cap in zip(
        box["whiskers"],
        box["caps"],
    ):
        whisker.set_color(palette[i+1])  # Set whisker color
        whisker.set_linewidth(1.5)  # Make whiskers thicker
        cap.set_color(palette[i+1])  # Set cap color
        cap.set_linewidth(1.5)
    plt.xticks([])
    # Customize the boxes
    for patch in box["boxes"]:
        patch.set_facecolor(faded_palette[i+1])
        patch.set_edgecolor(palette[i+1])  # Set vibrant edge color (no alpha)
        patch.set_linewidth(1.5)  # Edge thickness
        # patch.set_alpha(0.4)
        # Customize the median line
    for median in box["medians"]:
        median.set_color(palette[i+1])  # Set median line color
        median.set_linewidth(1.5)  # Make the line thicker
    for flier in box["fliers"]:
        flier.set_color(palette[i+1])


M = [50,100,150,250, 400]
N = [5,10,15,20,30, 50]

legend_elements = [
    Patch(facecolor=faded_palette[0], edgecolor=palette[0], label=f"{M[0]} chores"),
    Patch(facecolor=faded_palette[1], edgecolor=palette[1], label=f"{M[1]} chores"),
    Patch(facecolor=faded_palette[2], edgecolor=palette[2], label=f"{M[2]} chores"),
    Patch(facecolor=faded_palette[3], edgecolor=palette[3], label=f"{M[3]} chores"),
    Patch(facecolor=faded_palette[4], edgecolor=palette[4], label=f"{M[4]} chores"),
]
plt.legend(
    handles=legend_elements,
    # ncol=3,
    # bbox_to_anchor=(-0.049, 0.42),
    loc="upper left",
    fontsize=10,
)

# arr = np.concatenate([[1], np.arange(50, 1001, 50)])

plt.xticks(N, fontsize=8.2)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.12)
plt.xlabel("Number of Agents")
plt.ylabel("CPU time (seconds)")
plt.ylim([-50,2200])
plt.xlim([3,53])

plt.savefig(f"N_agents_runtime.jpg", dpi=300)
