import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plotly.colors import qualitative


palette = qualitative.Plotly
palette = [palette[i] for i in [1, 0, 2, 4, 3, 5]]
faded_palette = [color + "66" for color in palette]

df = pd.read_csv("experiments_ILP.csv")

num_chores_values = np.concatenate([np.arange(80, 101, 10), np.arange(150, 401, 50)])
agent_range = [5, 10, 15, 20]

plt.figure(figsize=(12, 3))
for i, n in enumerate(agent_range):
    runtime_values = []
    averages = []
    for num_chore_value in num_chores_values:
        filtered_df = df[
            (df["M"] == num_chore_value) & (df["model"] == "ILP") & (df["N"] == n)
        ]

        # filtered_df["seats"]=filtered_df["seats"].astype('float')
        runtime_values.append(filtered_df["CPU time"].values)
        averages.append(np.mean(filtered_df["CPU time"].values))
        # runtime_values.append(filtered_df["seats"].values)
        # averages.append(np.mean(filtered_df["seats"].values))
    print(averages)
    print(len(averages))

    plt.plot(num_chores_values, averages, color=palette[i])

    # plt.plot(num_students_values, averages, color=palette[i], marker="o")
    flierprops = dict(markeredgecolor=palette[i])
    box = plt.boxplot(
        runtime_values,
        positions=num_chores_values,
        patch_artist=True,
        widths=7,
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


legend_elements = [
    Patch(
        facecolor=faded_palette[0],
        edgecolor=palette[0],
        label=f"{agent_range[0]} agents",
    ),
    Patch(
        facecolor=faded_palette[1],
        edgecolor=palette[1],
        label=f"{agent_range[1]} agents",
    ),
    Patch(
        facecolor=faded_palette[2],
        edgecolor=palette[2],
        label=f"{agent_range[2]} agents",
    ),
    Patch(
        facecolor=faded_palette[3],
        edgecolor=palette[3],
        label=f"{agent_range[3]} agents",
    ),
    # Patch(facecolor=faded_palette[4], edgecolor=palette[4], label=f"{agent_range[4]} agents"),
]
plt.legend(
    handles=legend_elements,
    # ncol=3,
    # bbox_to_anchor=(-0.049, 0.42),
    loc="upper right",
    fontsize=12,
)

# arr = np.concatenate([[1], np.arange(50, 1001, 50)])

plt.xticks(num_chores_values, fontsize=8.2)
plt.tight_layout()
plt.subplots_adjust(bottom=0.07)
plt.subplots_adjust(left=0.08)
plt.xlabel("Number of Chores")
plt.ylabel("CPU time (seconds)")

plt.savefig(f"N_agents_runtime_30.jpg", dpi=300)
