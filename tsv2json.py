import argparse
import csv
import json
import os
from typing import Any, Dict, List


def convert_helpfulness_level(level: str) -> str:
    """
    Convert helpfulness level from CSV format to our format
    """
    mapping = {"HELPFUL": "helpful", "NOT_HELPFUL": "not helpful", "SOMEWHAT_HELPFUL": "somewhat helpful"}
    return mapping.get(level, "not helpful")


def get_helpful_reasons(row: Dict[str, Any]) -> List[str]:
    """
    Extract helpful reasons from a row
    """
    reasons = []

    # Map CSV columns to reason strings
    reason_mapping = {
        "helpfulClear": "Easy to understand",
        "helpfulGoodSources": "Cites high-quality sources",
        "helpfulAddressesClaim": "Directly addresses the post's claim",
        "helpfulImportantContext": "Provides important context",
        "helpfulUnbiasedLanguage": "Neutral or unbiased language",
    }

    # Check each reason field
    for field, reason in reason_mapping.items():
        if field in row and row[field] == "1":
            reasons.append(reason)

    # Add "Other" if specified
    if "helpfulOther" in row and row["helpfulOther"] == "1":
        reasons.append("Other")

    return reasons


def get_not_helpful_reasons(row: Dict[str, Any]) -> List[str]:
    """
    Extract not helpful reasons from a row
    """
    reasons = []

    # Map CSV columns to reason strings
    reason_mapping = {
        "notHelpfulIncorrect": "Incorrect information",
        "notHelpfulSourcesMissingOrUnreliable": "Sources not included or unreliable",
        "notHelpfulMissingKeyPoints": "Misses key points or irrelevant",
        "notHelpfulHardToUnderstand": "Typos or unclear language",
        "notHelpfulArgumentativeOrBiased": "Argumentative or biased language",
        "notHelpfulIrrelevantSources": "Sources do not support note",
        "notHelpfulOpinionSpeculation": "Opinion or speculation",
        "notHelpfulNoteNotNeeded": "Note not needed on this post",
        "notHelpfulSpamHarassmentOrAbuse": "Other",
    }

    # Check each reason field
    for field, reason in reason_mapping.items():
        if field in row and row[field] == "1":
            reasons.append(reason)

    # Add "Other" if specified
    if "notHelpfulOther" in row and row["notHelpfulOther"] == "1":
        reasons.append("Other")

    return reasons


def convert_csv_to_json(tsv_file: str, output_dir: str) -> None:
    """
    Convert TSV data to JSON files for tasks and groundtruth

    Args:
        tsv_file: Path to the TSV file
        output_dir: Directory to save the JSON files
    """
    # Create output directories
    tasks_dir = os.path.join(output_dir, "tasks")
    groundtruth_dir = os.path.join(output_dir, "groundtruth")

    os.makedirs(tasks_dir, exist_ok=True)
    os.makedirs(groundtruth_dir, exist_ok=True)

    # Read TSV file
    with open(tsv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        # Process only the first 20 rows
        for i, row in enumerate(list(reader)[:20]):
            # Create task JSON
            task_data = {"type": "community_note_evaluation", "user_id": row["raterParticipantId"], "note_id": row["noteId"]}

            # Save task JSON
            task_file = os.path.join(tasks_dir, f"task_{i}.json")
            with open(task_file, "w", encoding="utf-8") as f:
                json.dump(task_data, f, indent=2)

            # Determine helpfulness level
            helpfulness_level = convert_helpfulness_level(row.get("helpfulnessLevel", ""))

            # Get reasons based on helpfulness level
            reasons = []
            if helpfulness_level == "helpful":
                reasons = get_helpful_reasons(row)
            elif helpfulness_level == "not helpful":
                reasons = get_not_helpful_reasons(row)
            else:  # somewhat helpful
                # For somewhat helpful, we can include reasons from both categories
                helpful_reasons = get_helpful_reasons(row)
                not_helpful_reasons = get_not_helpful_reasons(row)
                reasons = helpful_reasons + not_helpful_reasons

            # If no reasons were found, add "Other"
            if not reasons:
                reasons = ["Other"]

            # Create groundtruth JSON
            groundtruth_data = {
                "evaluation": helpfulness_level,
                "reasons": reasons,
                "explanation": f"Evaluation based on user {row['raterParticipantId']}'s rating of note {row['noteId']}.",
            }

            # Save groundtruth JSON
            groundtruth_file = os.path.join(groundtruth_dir, f"groundtruth_{i}.json")
            with open(groundtruth_file, "w", encoding="utf-8") as f:
                json.dump(groundtruth_data, f, indent=2)

    print(f"TSVから最初の20行をJSONファイルに変換しました。出力先: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert TSV data to JSON for community note evaluation")
    parser.add_argument("--tsv_file", help="Path to the TSV file")
    parser.add_argument("--output_dir", default="./community_notes", help="Directory to save the JSON files")

    args = parser.parse_args()

    convert_csv_to_json(args.tsv_file, args.output_dir)


if __name__ == "__main__":
    main()
