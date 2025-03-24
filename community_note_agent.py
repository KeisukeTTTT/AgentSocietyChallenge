import json
import logging
import os
from typing import Any, Dict, List

from websocietysimulator.agent import SimulationAgent
from websocietysimulator.agent.modules.memory_modules import MemoryGenerative
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.llm import LLMBase

logging.basicConfig(level=logging.INFO)


class CommunityNoteEvaluationPlanning(PlanningBase):

    def __init__(self, llm):
        super().__init__(llm=llm)

    def __call__(self, task_description):
        self.plan = [
            {
                "description": "Get agent profile",
                "reasoning instruction": "Retrieve agent profile to understand evaluation perspective",
                "tool use instruction": {"agent_id": task_description.get("agent_id", "")},
            },
            {
                "description": "Get note content",
                "reasoning instruction": "Retrieve the content of the note to be evaluated",
                "tool use instruction": {"note_id": task_description.get("note_id", "")},
            },
            {
                "description": "Get agent's past evaluations",
                "reasoning instruction": "Retrieve how the agent has evaluated similar notes in the past",
                "tool use instruction": {"agent_id": task_description.get("agent_id", "")},
            },
            {
                "description": "Analyze evaluation patterns",
                "reasoning instruction": "Analyze the agent's evaluation patterns based on past evaluations",
                "tool use instruction": "Use memory module",
            },
            {
                "description": "Determine evaluation",
                "reasoning instruction": "Determine evaluation considering the agent's perspective and past evaluation patterns",
                "tool use instruction": "Use reasoning module",
            },
        ]
        return self.plan


class CommunityNoteEvaluationReasoning(ReasoningBase):
    """コミュニティノート評価のための推論モジュール"""

    def __init__(self, agent_profile, memory, llm):
        """推論モジュールの初期化"""
        super().__init__(profile_type_prompt="Community Note Evaluator", memory=memory, llm=llm)
        self.agent_profile = agent_profile

    def __call__(self, task_description: str, feedback: str = ""):
        """親クラスの__call__メソッドをオーバーライド"""
        examples, task_description = self.process_task_description(task_description)

        # エージェントの評価パターンに基づいてプロンプトを作成
        prompt = f"""あなたはコミュニティノート評価者です。以下のエージェントプロファイルと過去の評価パターンに基づいて、
与えられたノートを評価してください。

エージェントプロファイル:
{json.dumps(self.agent_profile, indent=2, ensure_ascii=False)}

評価対象のノート:
{task_description}

過去の評価パターンに基づいて、このノートが「helpful」「somewhat helpful」「not helpful」のどれに該当するか、
そしてその理由を選択してください。

評価結果をJSON形式で出力してください:
{{
  "evaluation": "helpful/somewhat helpful/not helpful",
  "reasons": ["理由1", "理由2", ...]
}}
"""

        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(messages=messages, temperature=0.2, max_tokens=1000)

        return reasoning_result


class CommunityNoteAgent(SimulationAgent):
    """コミュニティノート評価エージェント"""

    def __init__(self, llm: LLMBase, agent_profile: Dict[str, Any]):
        """評価エージェントの初期化"""
        super().__init__(llm=llm)
        self.memory = MemoryGenerative(llm=self.llm)
        self.agent_profile = agent_profile
        self.planning = CommunityNoteEvaluationPlanning(llm=self.llm)
        self.reasoning = CommunityNoteEvaluationReasoning(agent_profile=agent_profile, memory=self.memory, llm=self.llm)

        # 過去の評価をメモリに追加
        for evaluation in agent_profile.get("evaluations", []):
            evaluation_text = f"""
            Note ID: {evaluation.get('note_id', '')}
            Evaluation: {evaluation.get('evaluation', '')}
            Reasons: {', '.join(evaluation.get('reasons', []))}
            """
            self.memory(f"past_evaluation: {evaluation_text}")

    def workflow(self, note_content: str):
        try:
            # タスク情報を設定
            self.task = {"agent_id": self.agent_profile.get("agent_id", ""), "note_id": "current_note", "note_content": note_content}

            # プランを実行
            self.planning(task_description=self.task)

            # 推論を実行
            result = self.reasoning(task_description=f"Note content: {note_content}")

            # JSON形式の結果を解析
            try:
                evaluation = json.loads(result)
            except json.JSONDecodeError:
                # JSON解析に失敗した場合はテキスト解析を試みる
                evaluation = {}
                for line in result.split("\n"):
                    if "evaluation" in line.lower() and ":" in line:
                        evaluation["evaluation"] = line.split(":", 1)[1].strip().strip('"')
                    elif "reasons" in line.lower() and ":" in line:
                        reasons_text = line.split(":", 1)[1].strip()
                        if reasons_text.startswith("[") and reasons_text.endswith("]"):
                            try:
                                evaluation["reasons"] = json.loads(reasons_text)
                            except:
                                evaluation["reasons"] = [r.strip().strip("\"'") for r in reasons_text.strip("[]").split(",")]
                        else:
                            evaluation["reasons"] = [reasons_text]

            # 評価結果にエージェント情報を追加
            evaluation["agent_id"] = self.agent_profile.get("agent_id", "")
            evaluation["user_id"] = self.agent_profile.get("user_id", "")

            return evaluation

        except Exception as e:
            logging.error(f"Evaluation error: {e}")
            return {"error": str(e), "agent_id": self.agent_profile.get("agent_id", ""), "user_id": self.agent_profile.get("user_id", "")}


def load_agent_profiles(agents_dir: str) -> List[Dict[str, Any]]:
    agent_profiles = []
    for filename in os.listdir(agents_dir):
        if filename.startswith("agent_") and filename.endswith(".json"):
            with open(os.path.join(agents_dir, filename), "r", encoding="utf-8") as f:
                agent_profile = json.load(f)
                agent_profiles.append(agent_profile)
    return agent_profiles


def create_agents(llm: LLMBase, agents_dir: str) -> Dict[str, CommunityNoteAgent]:
    agent_profiles = load_agent_profiles(agents_dir)
    agents = {}
    for profile in agent_profiles:
        agent_id = profile.get("agent_id", "")
        if agent_id:
            agents[agent_id] = CommunityNoteAgent(llm=llm, agent_profile=profile)
    return agents
