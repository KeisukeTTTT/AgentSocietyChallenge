import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGIC_PATTERNS = {
    "Mitigation": "Mig",
    "Alternative": "Alt",
    "No Evidence": "No Evi",
    "Another True Cause": "ATC",
    "Missing Mechanism 1": "MM1",
    "Missing Mechanism 2": "MM2",
    "No Need to Address": "NNA",
    "Negative Effect due to y": "Neg eff",
    "Positive Effects of a Different Perspective from y 1": "Dif Per1",
    "Positive Effects of a Different Perspective from y 2": "Dif Per2",
}


def load_and_preprocess_data():
    with open("./misinformation_experiment/evaluations.json", encoding="utf-8") as f:
        evaluations = json.load(f)
    df = pd.DataFrame(evaluations)
    df = df[df["agent_type"] == "neutral"][["rating", "misinformation_type", "rebuttal_type"]]
    rating_map = {"helpful": 2, "somewhat helpful": 1, "not helpful": 0}
    df["rating_num"] = df["rating"].map(rating_map)
    return df


def plot_relationship_graph(df, central_rebuttal="Mitigation", lower_threshold=1.1, upper_threshold=1.5):
    grouped = df.groupby(["rebuttal_type", "misinformation_type"])["rating_num"].mean().reset_index()
    central_edges = grouped[grouped["rebuttal_type"] == central_rebuttal]
    filtered_edges = central_edges[(central_edges["rating_num"] > upper_threshold) | (central_edges["rating_num"] < lower_threshold)]
    connected_misinfo = filtered_edges["misinformation_type"].tolist()
    misinfo_list = central_edges["misinformation_type"].tolist()
    palette = plt.cm.tab10.colors
    misinfo_colors = {misinfo: palette[i % len(palette)] for i, misinfo in enumerate(misinfo_list)}
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    center_x, center_y = 0, 0
    center_marker_size = 120
    center_color = "#F05030"
    N = len(misinfo_list)
    radius = 9.0
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) if N > 0 else []
    misinfo_positions = {misinfo: (radius * np.cos(angle), radius * np.sin(angle)) for misinfo, angle in zip(misinfo_list, angles)}
    draw_edges(ax, filtered_edges, misinfo_positions, center_x, center_y, upper_threshold, lower_threshold, center_color)
    draw_nodes(ax, central_rebuttal, center_x, center_y, center_marker_size, center_color, misinfo_positions, misinfo_colors, connected_misinfo)
    plt.savefig(f"./misinformation_experiment/analysis/rebuttal_{LOGIC_PATTERNS[central_rebuttal].replace(' ', '_')}.png")


def draw_edges(ax, edges, positions, cx, cy, up_th, low_th, center_color):
    for _, row in edges.iterrows():
        misinfo = row["misinformation_type"]
        avg = row["rating_num"]
        x_target, y_target = positions[misinfo]
        if avg > up_th:
            props = {
                "arrowstyle": "Simple,head_length=2,head_width=2,tail_width=0.8",
                "facecolor": center_color,
                "edgecolor": "none",
                "mutation_scale": 20,
                "shrinkB": 45,
                "lw": 2,
                "linestyle": "-",
            }
        elif avg < low_th:
            props = {
                "arrowstyle": "-|>,head_length=0.5,head_width=0.2",
                "facecolor": center_color,
                "edgecolor": center_color,
                "mutation_scale": 20,
                "shrinkB": 45,
                "linestyle": "--",
            }
        else:
            continue
        ax.annotate("", xy=(x_target, y_target), xytext=(cx, cy), arrowprops=props, zorder=1)


def draw_nodes(ax, central_rebuttal, cx, cy, center_marker_size, center_color, positions, colors, connected):
    ax.plot(cx, cy, "o", markersize=center_marker_size, color=center_color, zorder=2)
    ax.text(cx, cy, LOGIC_PATTERNS[central_rebuttal], color="black", fontweight="bold", fontsize=20, ha="center", va="center", zorder=3)
    for misinfo, (x, y) in positions.items():
        alpha = 1.0 if misinfo in connected else 0.3
        ax.plot(x, y, "o", markersize=95, color=colors[misinfo], alpha=alpha, zorder=2)
        text_color = "black" if misinfo in connected else "gray"
        ax.text(x, y, misinfo.replace(" ", "\n"), color=text_color, fontweight="bold", fontsize=12, ha="center", va="center", zorder=3)


def main():
    df = load_and_preprocess_data()
    for central_rebuttal in df["rebuttal_type"].unique():
        plot_relationship_graph(df, central_rebuttal)


if __name__ == "__main__":
    main()
