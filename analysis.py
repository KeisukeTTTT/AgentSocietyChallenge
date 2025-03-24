import json
import os

import japanize_matplotlib  # noqa
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

font_size_ticks = 16
font_size_label = 16
font_size_title = 20


def save_heatmap(data, index, columns, values, title, filename, analysis_dir):
    """単一のヒートマップを画像ファイルとして保存する。"""
    plt.figure(figsize=(12, 8))
    pivot_table = data.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".2f", annot_kws={"size": font_size_ticks}, vmin=0, vmax=2)
    # plt.title(title, fontsize=16)

    # キャメルケースに変換
    x_label = "".join(word.capitalize() for word in columns.split("_"))
    y_label = "".join(word.capitalize() for word in index.split("_"))

    plt.xlabel(x_label, fontsize=font_size_label)
    plt.ylabel(y_label, fontsize=font_size_label)
    plt.xticks(fontsize=font_size_ticks, rotation=45)
    plt.yticks(fontsize=font_size_ticks)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, filename))
    plt.close()


def save_barplot(data, x, y, hue, title, filename, analysis_dir):
    """単一の棒グラフを画像ファイルとして保存する。"""
    plt.figure(figsize=(14, 6))
    sns.barplot(x=x, y=y, hue=hue, data=data)
    # plt.title(title, fontsize=font_size_title)

    # キャメルケースに変換
    x_label = "".join(word.capitalize() for word in x.split("_"))
    y_label = "".join(word.capitalize() for word in y.split("_"))

    plt.xlabel(x_label, fontsize=font_size_label)
    plt.ylabel(y_label, fontsize=font_size_label)
    plt.xticks(rotation=45, fontsize=font_size_ticks)
    plt.yticks(fontsize=font_size_ticks)
    plt.legend(fontsize=font_size_ticks)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, filename))
    plt.close()


def analyze_results(misinformations, rebuttals, evaluations, output_dir):
    """実験結果の分析処理（各種集計と可視化）"""
    # DataFrame に変換
    df = pd.DataFrame(evaluations)

    # 新評価形式対応：rating を数値に変換
    rating_map = {"helpful": 2, "somewhat helpful": 1, "not helpful": 0}
    if "rating" in df.columns:
        df["rating_score"] = df["rating"].map(rating_map)
    if "overall" not in df.columns and "rating_score" in df.columns:
        df["overall"] = df["rating_score"]

    # rebuttal_typeを略称に変換
    df["rebuttal_type"] = df["rebuttal_type"].map(lambda x: LOGIC_PATTERNS.get(x, x))

    # 分析結果保存用ディレクトリ作成
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # 1. 誤情報×反論タイプの効果分析
    misinfo_rebuttal_effectiveness = df.groupby(["misinformation_type", "rebuttal_type"])["overall"].mean().reset_index()
    best_rebuttals_by_misinfo = misinfo_rebuttal_effectiveness.loc[misinfo_rebuttal_effectiveness.groupby("misinformation_type")["overall"].idxmax()]
    best_rebuttals_by_misinfo.to_csv(os.path.join(analysis_dir, "best_rebuttals_by_misinfo_type.csv"), index=False)
    save_heatmap(
        misinfo_rebuttal_effectiveness,
        "misinformation_type",
        "rebuttal_type",
        "overall",
        "Effectiveness of Rebuttal Types by Misinformation Type",
        "misinfo_rebuttal_effectiveness.png",
        analysis_dir,
    )

    # 2. テーマ×反論タイプの効果分析
    theme_rebuttal_effectiveness = df.groupby(["theme", "rebuttal_type"])["overall"].mean().reset_index()
    best_rebuttals_by_theme = theme_rebuttal_effectiveness.loc[theme_rebuttal_effectiveness.groupby("theme")["overall"].idxmax()]
    best_rebuttals_by_theme.to_csv(os.path.join(analysis_dir, "best_rebuttals_by_theme.csv"), index=False)
    save_heatmap(
        theme_rebuttal_effectiveness,
        "theme",
        "rebuttal_type",
        "overall",
        "Effectiveness of Rebuttal Types by Theme",
        "theme_rebuttal_effectiveness.png",
        analysis_dir,
    )

    # 3. エージェントスタンスごとの反論評価のバイアス測定
    agent_bias = df.groupby(["agent_type", "rebuttal_type"])["overall"].mean().reset_index()
    save_barplot(
        agent_bias,
        "rebuttal_type",
        "overall",
        "agent_type",
        "Differences in Rebuttal Evaluation by Agent Stance",
        "agent_bias.png",
        analysis_dir,
    )

    # 4. 誤情報×反論の互換性マトリクス
    compatibility_matrix = misinfo_rebuttal_effectiveness.pivot(index="misinformation_type", columns="rebuttal_type", values="overall")
    compatibility_matrix.to_csv(os.path.join(analysis_dir, "misinfo_rebuttal_compatibility.csv"))

    # 新規追加分析: テーマ、エージェントタイプごとの偽情報タイプxリバッタルタイプのratingのヒートマップ(計6つ)
    for theme in df["theme"].unique():
        for agent_type in df["agent_type"].unique():
            theme_agent_df = df[(df["theme"] == theme) & (df["agent_type"] == agent_type)]
            if not theme_agent_df.empty:
                theme_agent_heatmap = theme_agent_df.groupby(["misinformation_type", "rebuttal_type"])["overall"].mean().reset_index()
                save_heatmap(
                    theme_agent_heatmap,
                    "misinformation_type",
                    "rebuttal_type",
                    "overall",
                    f"Rating Heatmap for {theme} - {agent_type}",
                    f"rating_heatmap_{theme.replace(' ', '_').lower()}_{agent_type}.png",
                    analysis_dir,
                )

    # 新規追加分析: テーマ、エージェントタイプごとの偽情報タイプxリバッタルタイプの分極度のヒートマップ(計6つ)
    for theme in df["theme"].unique():
        for agent_type in df["agent_type"].unique():
            theme_agent_df = df[(df["theme"] == theme) & (df["agent_type"] == agent_type)]
            if not theme_agent_df.empty:
                # 分極度（標準偏差）を計算
                polarization_heatmap = theme_agent_df.groupby(["misinformation_type", "rebuttal_type"])["overall"].std().reset_index()
                save_heatmap(
                    polarization_heatmap,
                    "misinformation_type",
                    "rebuttal_type",
                    "overall",
                    f"Polarization Heatmap for {theme} - {agent_type}",
                    f"polarization_heatmap_{theme.replace(' ', '_').lower()}_{agent_type}.png",
                    analysis_dir,
                )

    # 7. 評価の分散（Polarization）の分析
    polarization_data = df.groupby(["agent_type", "misinformation_type", "rebuttal_type"])["overall"].std().reset_index()
    save_barplot(
        polarization_data,
        "rebuttal_type",
        "overall",
        "agent_type",
        "Polarization of Evaluations by Agent Type, Misinformation Type, and Rebuttal Type",
        "polarization_analysis.png",
        analysis_dir,
    )

    # # 8. エージェントIDごとの評価傾向分析
    # agent_id_analysis = df.groupby(["agent_id", "rebuttal_type"])["overall"].mean().reset_index()
    # plt.figure(figsize=(16, 10))
    # sns.barplot(x="agent_id", y="overall", hue="rebuttal_type", data=agent_id_analysis)
    # plt.title("Evaluation Comparison by Agent ID", fontsize=16)
    # plt.xlabel("AgentId", fontsize=14)
    # plt.ylabel("OverallScore", fontsize=14)
    # plt.xticks(rotation=45, fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.legend(fontsize=12)
    # plt.tight_layout()
    # plt.savefig(os.path.join(analysis_dir, "agent_id_comparison.png"))
    # plt.close()

    # # 9. エージェント間の一貫性分析（変動係数）
    # agent_consistency = df.groupby(["agent_type", "misinformation_type", "rebuttal_type"])["overall"].agg(["mean", "std"]).reset_index()
    # agent_consistency["coefficient_of_variation"] = agent_consistency["std"] / agent_consistency["mean"]
    # plt.figure(figsize=(14, 8))
    # sns.boxplot(x="agent_type", y="coefficient_of_variation", data=agent_consistency)
    # plt.title("Evaluation Consistency by Agent Type", fontsize=16)
    # plt.xlabel("AgentType", fontsize=14)
    # plt.ylabel("CoefficientOfVariation", fontsize=14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.tight_layout()
    # plt.savefig(os.path.join(analysis_dir, "agent_consistency.png"))
    # plt.close()

    # # 10. エージェント間の評価相関とクラスタリング
    # agent_profiles = df.pivot_table(index=["misinformation_type", "rebuttal_type"], columns="agent_id", values="overall")
    # agent_correlation = agent_profiles.corr()
    # plt.figure(figsize=(15, 12))
    # sns.heatmap(agent_correlation, annot=True, cmap="viridis", fmt=".2f", annot_kws={"size": 12})
    # plt.title("Agent Evaluation Correlation", fontsize=16)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.tight_layout()
    # plt.savefig(os.path.join(analysis_dir, "agent_correlation.png"))
    # plt.close()

    # plt.figure(figsize=(16, 8))
    # hierarchy.dendrogram(hierarchy.linkage(agent_correlation, method="ward"), labels=agent_correlation.index, leaf_rotation=90)
    # plt.title("Hierarchical Clustering of Agent Evaluation Patterns", fontsize=16)
    # plt.xlabel("AgentId", fontsize=14)
    # plt.ylabel("Distance", fontsize=14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.tight_layout()
    # plt.savefig(os.path.join(analysis_dir, "agent_clustering.png"))
    # plt.close()

    # 11. 理由（reasons）の分布分析（該当する場合のみ）
    if "reasons" in df.columns:
        all_reasons = []
        for _, row in df.iterrows():
            if not isinstance(row.get("reasons", []), list):
                continue
            for reason in row.get("reasons", []):
                all_reasons.append({"agent_type": row["agent_type"], "rating": row.get("rating", ""), "reason": reason})
        if all_reasons:
            reasons_df = pd.DataFrame(all_reasons)
            reason_counts = reasons_df.groupby(["agent_type", "rating", "reason"]).size().reset_index(name="count")
            plt.figure(figsize=(16, 10))
            sns.barplot(x="reason", y="count", hue="rating", data=reason_counts)
            plt.title("Most Common Reasons Selected by Rating", fontsize=16)
            plt.xlabel("Reason", fontsize=14)
            plt.ylabel("Count", fontsize=14)
            plt.xticks(rotation=45, ha="right", fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, "reason_distribution.png"))
            plt.close()

    # 12. コミュニティ評価（rating）の分布分析（該当する場合のみ）
    if "rating" in df.columns:
        rating_distribution = df.groupby(["agent_type", "rating"]).size().reset_index(name="count")
        plt.figure(figsize=(12, 6))
        sns.barplot(x="agent_type", y="count", hue="rating", data=rating_distribution)
        plt.title("Distribution of Community Note Ratings by Agent Type", fontsize=16)
        plt.xlabel("AgentType", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, "rating_distribution.png"))
        plt.close()

    return {
        "best_rebuttals_by_misinfo": best_rebuttals_by_misinfo,
        "best_rebuttals_by_theme": best_rebuttals_by_theme,
        "agent_bias": agent_bias,
        # "agent_consistency": agent_consistency,
    }


if __name__ == "__main__":
    with open("./misinformation_experiment/misinformations.json", encoding="utf-8") as f:
        misinformations = json.load(f)
    with open("./misinformation_experiment/rebuttals.json", encoding="utf-8") as f:
        rebuttals = json.load(f)
    with open("./misinformation_experiment/evaluations.json", encoding="utf-8") as f:
        evaluations = json.load(f)

    analyze_results(misinformations, rebuttals, evaluations, "./misinformation_experiment")
