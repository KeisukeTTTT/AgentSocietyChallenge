import json
import os
import random
from collections import Counter

import pandas as pd
from tqdm import tqdm

# Set data directory
cn_data_dir = "/home/keisuke/work/TDAILab/NHK/community_note_analysis/data/community_note_dataset"
output_dir = "./community_notes"

# Create output directory
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "tasks"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "groundtruth"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "agents"), exist_ok=True)  # エージェント用ディレクトリ

# Load tsv files
print("Loading tsv files...")
notes = pd.read_csv(os.path.join(cn_data_dir, "notes-00000.tsv"), sep="\t", low_memory=False)
users = pd.read_csv(os.path.join(cn_data_dir, "userEnrollment-00000.tsv"), sep="\t", low_memory=False)
# 評価データは全て読み込む（head(1000)の制限を削除）
ratings = pd.read_csv(os.path.join(cn_data_dir, "ratings/ratings-00000.tsv"), sep="\t", low_memory=False)

# Delete deprecated columns
deprecated_columns = [
    "helpful",
    "notHelpful",
    "helpfulInformative",
    "helpfulEmpathetic",
    "helpfulUniqueContext",
    "notHelpfulOpinionSpeculationOrBias",
    "notHelpfulOutdated",
    "notHelpfulOffTopic",
]
ratings = ratings.drop(columns=deprecated_columns, errors="ignore")

# Convert user data
print("Converting user data...")
user_json = []
for _, row in tqdm(users.iterrows(), total=len(users)):
    user_json.append(
        {
            "user_id": str(row["participantId"]),
            "profile": f"modeling group: {row.get('modelingGroup', 'none')}, enrollment state: {row.get('enrollmentState', 'none')}",
        }
    )

# Convert note data
print("Converting note data...")
note_json = []
for _, row in tqdm(notes.iterrows(), total=len(notes)):
    note_json.append(
        {
            "note_id": str(row["noteId"]),
            "content": str(row.get("summary", "")),
            "classification": str(row.get("classification", "")),
        }
    )

# Convert evaluation data
print("Converting evaluation data...")
evaluation_json = []
for _, row in tqdm(ratings.iterrows(), total=len(ratings)):
    # Determine evaluation based on helpfulnessLevel
    helpfulness = row.get("helpfulnessLevel", "")
    if helpfulness == "HELPFUL":
        evaluation = "helpful"
        reasons = []
        if row.get("helpfulGoodSources", 0) == 1:
            reasons.append("Cites high-quality sources")
        if row.get("helpfulClear", 0) == 1:
            reasons.append("Easy to understand")
        if row.get("helpfulAddressesClaim", 0) == 1:
            reasons.append("Directly addresses the post's claim")
        if row.get("helpfulImportantContext", 0) == 1:
            reasons.append("Provides important context")
        if row.get("helpfulUnbiasedLanguage", 0) == 1:
            reasons.append("Neutral or unbiased language")
        if row.get("helpfulOther", 0) == 1:
            reasons.append("Other")
        if not reasons:
            reasons = ["Other"]
    elif helpfulness == "SOMEWHAT_HELPFUL":
        evaluation = "somewhat helpful"
        reasons = []
        # somewhat helpfulの場合は両方のカテゴリから理由を選択可能
        if row.get("helpfulGoodSources", 0) == 1:
            reasons.append("Cites high-quality sources")
        if row.get("notHelpfulSourcesMissingOrUnreliable", 0) == 1:
            reasons.append("Sources not included or unreliable")
        if row.get("helpfulClear", 0) == 1:
            reasons.append("Easy to understand")
        if row.get("notHelpfulHardToUnderstand", 0) == 1:
            reasons.append("Typos or unclear language")
        if not reasons:
            reasons = ["Other"]
    elif helpfulness == "NOT_HELPFUL":
        evaluation = "not helpful"
        reasons = []
        if row.get("notHelpfulSourcesMissingOrUnreliable", 0) == 1:
            reasons.append("Sources not included or unreliable")
        if row.get("notHelpfulIrrelevantSources", 0) == 1:
            reasons.append("Sources do not support note")
        if row.get("notHelpfulIncorrect", 0) == 1:
            reasons.append("Incorrect information")
        if row.get("notHelpfulOpinionSpeculation", 0) == 1:
            reasons.append("Opinion or speculation")
        if row.get("notHelpfulHardToUnderstand", 0) == 1:
            reasons.append("Typos or unclear language")
        if row.get("notHelpfulMissingKeyPoints", 0) == 1:
            reasons.append("Misses key points or irrelevant")
        if row.get("notHelpfulArgumentativeOrBiased", 0) == 1:
            reasons.append("Argumentative or biased language")
        if row.get("notHelpfulNoteNotNeeded", 0) == 1:
            reasons.append("Note not needed on this post")
        if row.get("notHelpfulOther", 0) == 1:
            reasons.append("Other")
        if not reasons:
            reasons = ["Other"]
    else:
        continue  # 評価レベルが不明な場合はスキップ

    evaluation_json.append(
        {
            "user_id": str(row["raterParticipantId"]),
            "note_id": str(row["noteId"]),
            "evaluation": evaluation,
            "reasons": reasons,
        }
    )

# JSONファイルの保存
print("Saving JSON files...")
with open(os.path.join(output_dir, "user.json"), "w", encoding="utf-8") as f:
    json.dump(user_json, f, ensure_ascii=False, indent=2)

with open(os.path.join(output_dir, "note.json"), "w", encoding="utf-8") as f:
    json.dump(note_json, f, ensure_ascii=False, indent=2)

with open(os.path.join(output_dir, "evaluation.json"), "w", encoding="utf-8") as f:
    json.dump(evaluation_json, f, ensure_ascii=False, indent=2)

# 評価実績が10件以上あるユーザーを特定
print("Identifying users with 10+ evaluations...")
user_evaluation_counts = Counter([eval_data["user_id"] for eval_data in evaluation_json])
qualified_users = [user_id for user_id, count in user_evaluation_counts.items() if count >= 10]

print(f"Found {len(qualified_users)} users with 10+ evaluations")

# 10人のユーザーをランダムに選択（もし10人以上いれば）
num_agents = 20
if len(qualified_users) >= num_agents:
    selected_users = random.sample(qualified_users, num_agents)
else:
    selected_users = qualified_users
    print(f"Warning: Only {len(qualified_users)} users have 10+ evaluations")

# 選択されたユーザーの評価データを収集
print("Collecting evaluation data for selected users...")
user_evaluations = {}
for user_id in selected_users:
    user_evaluations[user_id] = [eval_data for eval_data in evaluation_json if eval_data["user_id"] == user_id]

# 各ユーザーのエージェントプロファイルを作成
print("Creating agent profiles...")
agent_profiles = []
for i, user_id in enumerate(selected_users, 1):
    # ユーザープロファイル情報を取得
    user_profile = next((user for user in user_json if user["user_id"] == user_id), {"profile": "unknown"})

    # 評価パターンを分析
    evaluations = user_evaluations[user_id]
    evaluation_types = Counter([eval_data["evaluation"] for eval_data in evaluations])
    reason_counter = Counter()
    for eval_data in evaluations:
        for reason in eval_data["reasons"]:
            reason_counter[reason] += 1

    top_reasons = [reason for reason, _ in reason_counter.most_common(5)]

    # エージェントプロファイルを作成
    agent_profile = {
        "agent_id": f"agent_{i}",
        "user_id": user_id,
        "profile": user_profile["profile"],
        "evaluation_count": len(evaluations),
        "evaluation_pattern": {
            "helpful": evaluation_types.get("helpful", 0),
            "somewhat_helpful": evaluation_types.get("somewhat helpful", 0),
            "not_helpful": evaluation_types.get("not helpful", 0),
        },
        "top_reasons": top_reasons,
        "evaluations": evaluations,
    }

    agent_profiles.append(agent_profile)

    # 個別のエージェントファイルを保存
    with open(os.path.join(output_dir, "agents", f"agent_{i}.json"), "w", encoding="utf-8") as f:
        json.dump(agent_profile, f, ensure_ascii=False, indent=2)

# エージェントプロファイルの概要を保存
with open(os.path.join(output_dir, "agent_profiles.json"), "w", encoding="utf-8") as f:
    json.dump(
        [
            {
                "agent_id": profile["agent_id"],
                "user_id": profile["user_id"],
                "profile": profile["profile"],
                "evaluation_count": profile["evaluation_count"],
                "evaluation_pattern": profile["evaluation_pattern"],
                "top_reasons": profile["top_reasons"],
            }
            for profile in agent_profiles
        ],
        f,
        ensure_ascii=False,
        indent=2,
    )

# タスクとグラウンドトゥルースの作成
print("Creating tasks and groundtruth...")
# 評価データがある組み合わせを抽出
valid_combinations = []
for eval_data in evaluation_json:
    valid_combinations.append((eval_data["user_id"], eval_data["note_id"], eval_data))

# ランダムに選択（最大100件）
num_tasks = min(100, len(valid_combinations))
selected_combinations = random.sample(valid_combinations, num_tasks)

# テスト用に選択された評価を記録
test_evaluations = set()
for user_id, note_id, _ in selected_combinations:
    test_evaluations.add((user_id, note_id))

# トレーニング用の評価データを作成（テストデータを除外）
training_evaluation_json = [eval_data for eval_data in evaluation_json if (eval_data["user_id"], eval_data["note_id"]) not in test_evaluations]

# トレーニング用の評価データを保存
with open(os.path.join(output_dir, "training_evaluation.json"), "w", encoding="utf-8") as f:
    json.dump(training_evaluation_json, f, ensure_ascii=False, indent=2)

for i, (user_id, note_id, eval_data) in enumerate(selected_combinations, 1):
    # タスクファイル
    task_data = {"type": "community_note_evaluation", "user_id": user_id, "note_id": note_id}
    with open(os.path.join(output_dir, "tasks", f"task_{i}.json"), "w", encoding="utf-8") as f:
        json.dump(task_data, f, ensure_ascii=False, indent=2)

    # グラウンドトゥルースファイル
    groundtruth_data = {"evaluation": eval_data["evaluation"], "reasons": eval_data["reasons"]}
    with open(os.path.join(output_dir, "groundtruth", f"groundtruth_{i}.json"), "w", encoding="utf-8") as f:
        json.dump(groundtruth_data, f, ensure_ascii=False, indent=2)

print(f"Data conversion completed. {num_tasks} tasks created. Training data contains {len(training_evaluation_json)} evaluations.")
print(f"Created {len(agent_profiles)} agent profiles based on users with 10+ evaluations.")
