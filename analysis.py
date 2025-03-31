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

FONT_SIZE_TICKS = 16
FONT_SIZE_LABEL = 16
FONT_SIZE_TITLE = 20


def save_heatmap(data, index, columns, values, filename, analysis_dir):
    """単一のヒートマップを画像ファイルとして保存する。"""
    plt.figure(figsize=(12, 8))
    pivot_table = data.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".2f", annot_kws={"size": FONT_SIZE_TICKS}, vmin=0, vmax=2)
    # plt.title(title, fontsize=16)

    # キャメルケースに変換
    x_label = "".join(word.capitalize() for word in columns.split("_"))
    y_label = "".join(word.capitalize() for word in index.split("_"))

    plt.xlabel(x_label, fontsize=FONT_SIZE_LABEL)
    plt.ylabel(y_label, fontsize=FONT_SIZE_LABEL)
    plt.xticks(fontsize=FONT_SIZE_TICKS, rotation=45)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, filename))
    plt.close()


def save_barplot(data, x, y, hue, filename, analysis_dir):
    """単一の棒グラフを画像ファイルとして保存する。"""
    plt.figure(figsize=(14, 6))
    sns.barplot(x=x, y=y, hue=hue, data=data)
    # plt.title(title, fontsize=FONT_SIZE_TITLE)

    # キャメルケースに変換
    x_label = "".join(word.capitalize() for word in x.split("_"))
    y_label = "".join(word.capitalize() for word in y.split("_"))

    plt.xlabel(x_label, fontsize=FONT_SIZE_LABEL)
    plt.ylabel(y_label, fontsize=FONT_SIZE_LABEL)
    plt.xticks(rotation=45, fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.legend(fontsize=FONT_SIZE_TICKS)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, filename))
    plt.close()


# 1. 誤情報×反論タイプの効果分析
def analyze_misinfo_rebuttal_effectiveness(df, analysis_dir):
    misinfo_rebuttal_effectiveness = df.groupby(["misinformation_type", "rebuttal_type"])["rating_score"].mean().reset_index()
    save_heatmap(
        misinfo_rebuttal_effectiveness,
        "misinformation_type",
        "rebuttal_type",
        "rating_score",
        "misinfo_rebuttal_effectiveness.png",
        analysis_dir,
    )


# 2. テーマ×反論タイプの効果分析
def analyze_theme_rebuttal_effectiveness(df, analysis_dir):
    theme_rebuttal_effectiveness = df.groupby(["theme", "rebuttal_type"])["rating_score"].mean().reset_index()
    save_heatmap(
        theme_rebuttal_effectiveness,
        "theme",
        "rebuttal_type",
        "rating_score",
        "theme_rebuttal_effectiveness.png",
        analysis_dir,
    )


# 3. エージェントスタンスごとの反論評価のバイアス測定
def analyze_agent_rebuttal_bias(df, analysis_dir):
    agent_bias = df.groupby(["agent_type", "rebuttal_type"])["rating_score"].std().reset_index()
    save_barplot(
        agent_bias,
        "rebuttal_type",
        "rating_score",
        "agent_type",
        "agent_bias.png",
        analysis_dir,
    )


# 4. 誤情報×反論の互換性マトリクス
def analyze_misinfo_rebuttal_compatibility(df, analysis_dir):
    misinfo_rebuttal_effectiveness = df.groupby(["misinformation_type", "rebuttal_type"])["rating_score"].mean().reset_index()
    compatibility_matrix = misinfo_rebuttal_effectiveness.pivot(index="misinformation_type", columns="rebuttal_type", values="rating_score")
    compatibility_matrix.to_csv(os.path.join(analysis_dir, "misinfo_rebuttal_compatibility.csv"))

    # 新規追加分析: テーマ、エージェントタイプごとの偽情報タイプxリバッタルタイプのratingのヒートマップ(計6つ)
    for theme in df["theme"].unique():
        for agent_type in df["agent_type"].unique():
            theme_agent_df = df[(df["theme"] == theme) & (df["agent_type"] == agent_type)]
            if not theme_agent_df.empty:
                theme_agent_heatmap = theme_agent_df.groupby(["misinformation_type", "rebuttal_type"])["rating_score"].mean().reset_index()
                save_heatmap(
                    theme_agent_heatmap,
                    "misinformation_type",
                    "rebuttal_type",
                    "rating_score",
                    f"rating_heatmap_{theme.replace(' ', '_').lower()}_{agent_type}.png",
                    analysis_dir,
                )

    # 新規追加分析: テーマ、エージェントタイプごとの偽情報タイプxリバッタルタイプの分極度のヒートマップ(計6つ)
    for theme in df["theme"].unique():
        for agent_type in df["agent_type"].unique():
            theme_agent_df = df[(df["theme"] == theme) & (df["agent_type"] == agent_type)]
            if not theme_agent_df.empty:
                # 分極度（標準偏差）を計算
                polarization_heatmap = theme_agent_df.groupby(["misinformation_type", "rebuttal_type"])["rating_score"].std().reset_index()
                save_heatmap(
                    polarization_heatmap,
                    "misinformation_type",
                    "rebuttal_type",
                    "rating_score",
                    f"polarization_heatmap_{theme.replace(' ', '_').lower()}_{agent_type}.png",
                    analysis_dir,
                )


# 7. 評価の分散（Polarization）の分析
def analyze_evaluation_polarization(df, analysis_dir):
    polarization_data = df.groupby(["agent_type", "misinformation_type", "rebuttal_type"])["rating_score"].std().reset_index()
    save_barplot(
        polarization_data,
        "rebuttal_type",
        "rating_score",
        "agent_type",
        "polarization_analysis.png",
        analysis_dir,
    )


# 11. 理由（reasons）の分布分析（該当する場合のみ）
def analyze_reason_distribution(df, analysis_dir):
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
            plt.xlabel("Reason", fontsize=14)
            plt.ylabel("Count", fontsize=14)
            plt.xticks(rotation=45, ha="right", fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, "reason_distribution.png"))
            plt.close()


# 12. コミュニティ評価（rating）の分布分析（該当する場合のみ）
def analyze_rating_distribution(df, analysis_dir):
    if "rating" in df.columns:
        rating_distribution = df.groupby(["agent_type", "rating"]).size().reset_index(name="count")
        plt.figure(figsize=(12, 6))
        sns.barplot(x="agent_type", y="count", hue="rating", data=rating_distribution)
        plt.xlabel("AgentType", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, "rating_distribution.png"))
        plt.close()


def analyze_results(evaluations, output_dir):
    """実験結果の分析処理（各種集計と可視化）"""
    # DataFrame に変換
    df = pd.DataFrame(evaluations)

    # 新評価形式対応：rating を数値に変換
    rating_map = {"helpful": 2, "somewhat helpful": 1, "not helpful": 0}
    if "rating" in df.columns:
        df["rating_score"] = df["rating"].map(rating_map)

    # rebuttal_typeを略称に変換
    df["rebuttal_type"] = df["rebuttal_type"].map(lambda x: LOGIC_PATTERNS.get(x, x))

    # 分析結果保存用ディレクトリ作成
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    analyze_misinfo_rebuttal_effectiveness(df, analysis_dir)
    analyze_theme_rebuttal_effectiveness(df, analysis_dir)
    analyze_agent_rebuttal_bias(df, analysis_dir)
    analyze_misinfo_rebuttal_compatibility(df, analysis_dir)
    analyze_evaluation_polarization(df, analysis_dir)
    analyze_reason_distribution(df, analysis_dir)
    analyze_rating_distribution(df, analysis_dir)


if __name__ == "__main__":
    with open("./misinformation_experiment/evaluations.json", encoding="utf-8") as f:
        evaluations = json.load(f)

    analyze_results(evaluations, "./misinformation_experiment")
