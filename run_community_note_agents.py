import json
import logging
import os
from typing import Any, Dict, List

from community_note_agent import CommunityNoteAgent, create_agents
from websocietysimulator.llm import OpenAILLM

logging.basicConfig(level=logging.INFO)


def run_evaluation(agents: Dict[str, CommunityNoteAgent], notes: List[Dict[str, Any]], num_notes: int = 5):
    import random

    selected_notes = random.sample(notes, min(num_notes, len(notes)))

    results = {agent_id: [] for agent_id in agents.keys()}

    # 各ノートに対して全エージェントで評価
    for note in selected_notes:
        note_id = note.get("note_id", "")
        note_content = note.get("content", "")

        print(f"Evaluating note {note_id}...")

        for agent_id, agent in agents.items():
            print(f"  Agent {agent_id} evaluating...")
            evaluation = agent.workflow(note_content)
            evaluation["note_id"] = note_id
            results[agent_id].append(evaluation)

    return results


def main():
    # 出力ディレクトリ
    output_dir = "./community_notes"
    agents_dir = os.path.join(output_dir, "agents")

    # LLMの初期化
    llm = OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

    # エージェントの作成
    agents = create_agents(llm=llm, agents_dir=agents_dir)
    print(f"Created {len(agents)} agents")

    # ノートデータの読み込み
    with open(os.path.join(output_dir, "note.json"), "r", encoding="utf-8") as f:
        notes = json.load(f)
    print(f"Loaded {len(notes)} notes")

    # 評価の実行
    results = run_evaluation(agents=agents, notes=notes, num_notes=5)

    # 結果の保存
    with open(os.path.join(output_dir, "agent_evaluations.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Evaluation completed. Results saved to agent_evaluations.json")


if __name__ == "__main__":
    main()
