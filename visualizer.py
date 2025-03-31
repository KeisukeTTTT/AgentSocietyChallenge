import json
import os

import japanize_matplotlib  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

FONT_SIZE_TICKS = 16
FONT_SIZE_LABEL = 16
FONT_SIZE_TITLE = 20

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

# Categorization of misinformation types with concise names
MISINFO_CATEGORIES = {
    "Evidence Issues": [
        "Faulty Generalization",
        "False Causality",
        "False Dilemma",
        "Deductive Fallacy",
    ],
    "Structure Issues": [
        "Circular Claim",
        "Equivocation",
        "Fallacy of Extension",
        "Fallacy of Relevance",
    ],
    "Persuasion Issues": [
        "Ad Hominem",
        "Ad Populum",
        "Appeal to Emotion",
        "Fallacy of Credibility",
        "Intentional Fallacy",
    ],
}

# Categorization of rebuttal types with concise names
# REBUTTAL_CATEGORIES = {
#     "Causality": ["Mig", "Alt", "No Evi", "ATC", "MM1", "MM2"],
#     "Value": ["NNA", "Neg eff"],
#     "Conclusion": ["Dif Per1", "Dif Per2"],
# }

# REBUTTAL_CATEGORIES = {
#     "LR": ["Mig", "ATC", "MM1", "MM2"],  # Logical Refutation: 論理的反駁
#     "AE": ["Alt"],  # Alternative Explanation: 代替説明
#     "EP": ["No Evi"],  # Evidence Presentation: 証拠提示
#     "FP": ["NNA", "Neg eff"],  # Fact Presentation: 事実提示
#     "DPE": ["Dif Per1", "Dif Per2"],  # Different Perspective Effects: 異なる視点の効果
# }

REBUTTAL_CATEGORIES = {
    "Mig": "Logical Refutation",  # Logical Refutation: 論理的反駁
    "ATC": "Logical Refutation",
    "MM1": "Logical Refutation",
    "MM2": "Logical Refutation",
    "Alt": "Alternative Explanation",  # Alternative Explanation: 代替説明
    "No Evi": "Evidence Presentation",  # Evidence Presentation: 証拠提示
    "NNA": "Fact Presentation",  # Fact Presentation: 事実提示
    "Neg eff": "Fact Presentation",
    "Dif Per1": "Different Perspective Effects",  # Different Perspective Effects: 異なる視点の効果
    "Dif Per2": "Different Perspective Effects",
}


def load_and_preprocess_data():
    with open("./misinformation_experiment/evaluations.json", encoding="utf-8") as f:
        evaluations = json.load(f)
    df = pd.DataFrame(evaluations)
    # df = df[df["agent_type"] == "neutral"][["rating", "misinformation_type", "rebuttal_type"]]
    rating_map = {"helpful": 2, "somewhat helpful": 1, "not helpful": 0}
    df["rating_score"] = df["rating"].map(rating_map)
    df["rebuttal_type"] = df["rebuttal_type"].map(lambda x: LOGIC_PATTERNS.get(x, x))
    return df


def plot_relationship_graph(df, central_rebuttal="Mitigation", lower_threshold=1.1, upper_threshold=1.5):
    grouped = df.groupby(["rebuttal_type", "misinformation_type"])["rating_score"].mean().reset_index()
    central_edges = grouped[grouped["rebuttal_type"] == central_rebuttal]
    filtered_edges = central_edges[(central_edges["rating_score"] > upper_threshold) | (central_edges["rating_score"] < lower_threshold)]
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
    plt.savefig(f"./misinformation_experiment/analysis/rebuttal_{central_rebuttal.replace(' ', '_')}.png")


def draw_edges(ax, edges, positions, cx, cy, up_th, low_th, center_color):
    for _, row in edges.iterrows():
        misinfo = row["misinformation_type"]
        avg = row["rating_score"]
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
    ax.text(cx, cy, central_rebuttal, color="black", fontweight="bold", fontsize=20, ha="center", va="center", zorder=3)
    for misinfo, (x, y) in positions.items():
        alpha = 1.0 if misinfo in connected else 0.3
        ax.plot(x, y, "o", markersize=95, color=colors[misinfo], alpha=alpha, zorder=2)
        text_color = "black" if misinfo in connected else "gray"
        ax.text(x, y, misinfo.replace(" ", "\n"), color=text_color, fontweight="bold", fontsize=12, ha="center", va="center", zorder=3)


def analyze_misinfo_difficulty(df, save_dir):
    pivot_df = df.pivot_table(index="misinformation_type", columns="agent_type", values="rating_score", aggfunc="mean")

    if "neutral" in pivot_df.columns and "pro_misinformation" in pivot_df.columns:
        pivot_df["difference"] = pivot_df["neutral"] - pivot_df["pro_misinformation"]

    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f", vmin=0, vmax=2)
    plt.xlabel("Agent Type")
    plt.ylabel("Misinformation Type")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "misinfo_difficulty.png"))


def analyze_rebuttal_difficulty(df, save_dir):
    pivot_df = df.pivot_table(index="rebuttal_type", columns="agent_type", values="rating_score", aggfunc="mean")

    if "neutral" in pivot_df.columns and "pro_misinformation" in pivot_df.columns:
        pivot_df["difference"] = pivot_df["neutral"] - pivot_df["pro_misinformation"]

    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f", vmin=0, vmax=2)
    plt.xlabel("Agent Type")
    plt.ylabel("Rebuttal Type")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rebuttal_difficulty.png"))


def analyze_categorized_heatmaps(df, save_dir):
    """
    Creates a single 3x3 heatmap showing relationships between
    misinformation categories and rebuttal categories.
    """

    # Create category mapping for misinformation types
    misinfo_to_category = {}
    for category, types in MISINFO_CATEGORIES.items():
        for misinfo_type in types:
            misinfo_to_category[misinfo_type] = category

    # Create category mapping for rebuttal types
    rebuttal_to_category = {}
    for category, types in REBUTTAL_CATEGORIES.items():
        for rebuttal_type in types:
            rebuttal_to_category[rebuttal_type] = category

    # Add category information
    df_with_categories = df.copy()
    df_with_categories["misinfo_category"] = df_with_categories["misinformation_type"].map(misinfo_to_category)
    df_with_categories["rebuttal_category"] = df_with_categories["rebuttal_type"].map(rebuttal_to_category)

    # Calculate mean rating scores for each category combination
    category_pivot = df_with_categories.pivot_table(index="misinfo_category", columns="rebuttal_category", values="rating_score", aggfunc="mean")

    # Ensure the order of categories is preserved
    ordered_misinfo = sorted(MISINFO_CATEGORIES.keys())
    ordered_rebuttal = sorted(REBUTTAL_CATEGORIES.keys())

    category_pivot = category_pivot.reindex(index=ordered_misinfo, columns=ordered_rebuttal)

    # Create a 3x3 heatmap with improved formatting
    plt.figure(figsize=(12, 10))

    # Use a custom colormap for better visibility
    cmap = sns.color_palette("YlGnBu", as_cmap=True)

    # Create the heatmap with annotations
    sns.heatmap(
        category_pivot,
        annot=True,
        cmap=cmap,
        fmt=".2f",
        vmin=0,
        vmax=2,
    )

    # Customize the appearance
    plt.title("Effectiveness of Rebuttal Categories Against Misinformation Categories", fontsize=16, pad=20)

    # Make category names more readable by adjusting text size and labels
    plt.xlabel("Rebuttal Category", fontsize=14, labelpad=10)
    plt.ylabel("Misinformation Category", fontsize=14, labelpad=10)

    # Adjust tick label size
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(save_dir, "category_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()


def analyze_rebuttals_by_misinfo_heatmap(df, save_dir):
    """
    Creates a heatmap showing relationships between individual misinformation types
    and categorized rebuttal types.
    """

    # Create category mapping for rebuttal types
    rebuttal_to_category = {}
    for category, types in REBUTTAL_CATEGORIES.items():
        for rebuttal_type in types:
            rebuttal_to_category[rebuttal_type] = category

    # Add category information
    df_with_categories = df.copy()
    df_with_categories["rebuttal_category"] = df_with_categories["rebuttal_type"].map(rebuttal_to_category)

    # Calculate mean rating scores for each misinformation type and rebuttal category
    pivot_df = df_with_categories.pivot_table(index="misinformation_type", columns="rebuttal_category", values="rating_score", aggfunc="mean")

    # Ensure the order of rebuttal categories is preserved
    ordered_rebuttal = sorted(REBUTTAL_CATEGORIES.keys())
    pivot_df = pivot_df.reindex(columns=ordered_rebuttal)

    # Create custom sorting metric based on compatibility with rebuttal categories
    # First, ensure all three categories exist in columns (handling missing categories)
    for cat in ordered_rebuttal:
        if cat not in pivot_df.columns:
            pivot_df[cat] = np.nan

    # Create the heatmap
    plt.figure(figsize=(10, 12))

    # Use a custom colormap for better visibility
    cmap = sns.color_palette("YlGnBu", as_cmap=True)

    # Create the heatmap with annotations
    sns.heatmap(pivot_df, annot=True, cmap=cmap, fmt=".2f", vmin=0, vmax=2, linewidths=0.5, linecolor="gray", cbar_kws={"label": "Mean Rating Score"})

    # Customize the appearance
    plt.title("Effectiveness of Rebuttal Categories Against Misinformation Types", fontsize=16, pad=20)

    # Adjust labels and text size
    plt.xlabel("Rebuttal Category", fontsize=14, labelpad=10)
    plt.ylabel("Misinformation Type", fontsize=14, labelpad=10)

    # Adjust tick label size
    plt.xticks(fontsize=12, rotation=45, ha="right")
    plt.yticks(fontsize=12, rotation=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(save_dir, "rebuttal_category_by_misinfo_type.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_rebuttal_difficulty_by_misinfo_type(df, save_dir):

    df_ = df.groupby(["misinformation_type", "rebuttal_type", "agent_type"])["rating_score"].mean().reset_index()
    pivot_df = df_.pivot_table(index="misinformation_type", columns="agent_type", values="rating_score", aggfunc="max")

    plt.figure(figsize=(10, 10))
    plt.scatter(pivot_df["neutral"], pivot_df["neutral"] - pivot_df["pro_misinformation"])

    # テキストのリストを作成
    texts = []
    for i, type_name in enumerate(pivot_df.index):
        x = pivot_df["neutral"].iloc[i]
        y = pivot_df["neutral"].iloc[i] - pivot_df["pro_misinformation"].iloc[i]
        texts.append(plt.text(x, y, type_name))

    adjust_text(texts)

    plt.xlabel("neutral", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("difference", fontsize=FONT_SIZE_LABEL)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "misinfo_difficulty_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_rebuttal_difficulty_by_rebuttal_type(df, save_dir):
    df_ = df.groupby(["misinformation_type", "rebuttal_type", "agent_type"])["rating_score"].mean().reset_index()
    pivot_df = df_.pivot_table(index="rebuttal_type", columns="agent_type", values="rating_score", aggfunc="mean")
    # pivot_df["rebuttal_type"] = pivot_df.index.map(LOGIC_PATTERNS)
    pivot_df["group"] = pivot_df["rebuttal_type"].map(REBUTTAL_CATEGORIES)
    pivot_df = pivot_df.sort_values(by="neutral", ascending=False)
    # グループごとに色を定義
    group_colors = {
        "Logical Refutation": "red",
        "Alternative Explanation": "blue",
        "Evidence Presentation": "green",
        "Fact Presentation": "purple",
        "Different Perspective Effects": "orange",
    }

    # グループに基づいて色のリストを作成
    colors = [group_colors[group] for group in pivot_df["group"]]

    plt.figure(figsize=(10, 10))
    # 色をcmapではなく、c引数で指定
    plt.scatter(pivot_df["neutral"], pivot_df["neutral"] - pivot_df["pro_misinformation"], c=colors)

    # 凡例を追加
    for group, color in group_colors.items():
        plt.plot([], [], "o", color=color, label=group)
    plt.legend()

    # 各ポイントにmisinformation_typeをラベルとして追加
    for i, type_name in enumerate(pivot_df.index):
        plt.annotate(
            type_name, (pivot_df["neutral"][i], pivot_df["neutral"][i] - pivot_df["pro_misinformation"][i]), xytext=(5, 5), textcoords="offset points"
        )

    plt.xlabel("neutral", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("difference", fontsize=FONT_SIZE_LABEL)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rebuttal_difficulty_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df = load_and_preprocess_data()
    save_dir = "./misinformation_experiment/analysis"

    # 既存の分析関数を呼び出し
    analyze_misinfo_difficulty(df, save_dir)
    analyze_rebuttal_difficulty(df, save_dir)

    # 新しいカテゴリ別分析関数を呼び出し
    analyze_categorized_heatmaps(df, save_dir)
    analyze_rebuttals_by_misinfo_heatmap(df, save_dir)
    plot_rebuttal_difficulty_by_misinfo_type(df, save_dir)
    plot_rebuttal_difficulty_by_rebuttal_type(df, save_dir)
    for central_rebuttal in df["rebuttal_type"].unique():
        plot_relationship_graph(df, central_rebuttal)


if __name__ == "__main__":
    main()
