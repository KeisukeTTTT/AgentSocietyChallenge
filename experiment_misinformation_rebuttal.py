import concurrent.futures
import json
import logging
import os

from tqdm import tqdm

from prompts import EVALUATION_PROMPT, MISINFORMATION_GENERATION_PROMPT, REBUTTAL_DESCRIPTIONS, REBUTTAL_GENERATION_PROMPT
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.agent.modules.memory_modules import MemoryGenerative
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.llm import LLMBase, OpenAILLM
from websocietysimulator.tools import CacheInteractionTool

# プロンプト変数を取得
MISINFORMATION_GENERATION_PROMPT = MISINFORMATION_GENERATION_PROMPT
REBUTTAL_GENERATION_PROMPT = REBUTTAL_GENERATION_PROMPT
EVALUATION_PROMPT = EVALUATION_PROMPT
REBUTTAL_DESCRIPTIONS = REBUTTAL_DESCRIPTIONS

logging.basicConfig(level=logging.WARNING)

# Experiment settings
THEMES = ["Gun Control", "Partisan Alignment", "Abortion Ban"]
# THEMES = ["Gun Control"]

# MISINFORMATION_TYPES = [
#     "Satire/parody",
#     "Misleading content",
#     "False connection",
#     "False context",
#     "Imposter content",
#     "Manipulated content",
#     "Fabricated content",
# ]

MISINFORMATION_TYPES = [
    "Intentional Fallacy",
    "Appeal to Emotion",
    "Faulty Generalization",
    "Fallacy of Credibility",
    "Ad Hominem",
    "Fallacy of Relevance",
    "Deductive Fallacy",
    "False Causality",
    "Fallacy of Extension",
    "Ad Populum",
    "False Dilemma",
    "Equivocation",
    "Circular Claim",
]

REBUTTAL_TYPES = [
    "Mitigation",
    "Alternative",
    "No Evidence",
    "Another True Cause",
    "Missing Mechanism 1",
    "Missing Mechanism 2",
    "No Need to Address",
    "Negative Effect due to y",
    "Positive Effects of a Different Perspective from y 1",
    "Positive Effects of a Different Perspective from y 2",
]

LOGICAL_FALLACIES = {
    "Faulty Generalization": "Drawing a broad conclusion from a small sample or limited evidence",
    "Ad Hominem": "Attacking the person instead of addressing their argument",
    "Ad Populum": "Appealing to widespread belief, majority opinion, or popularity as validation",
    "False Causality": "Assuming causation based on correlation between events",
    "Circular Claim": "Using the conclusion as a premise to support the conclusion",
    "Appeal to Emotion": "Using emotion rather than logic to support an argument",
    "Fallacy of Relevance": "Introducing irrelevant information to divert from the main argument",
    "Deductive Fallacy": "Errors in the logical structure of an argument",
    "Intentional Fallacy": "Deliberately misleading arguments designed to win without legitimate evidence",
    "Fallacy of Extension": "Distorting or exaggerating an opponent's position to make it easier to attack",
    "False Dilemma": "Presenting only two options when more exist",
    "Fallacy of Credibility": "Inappropriately appealing to authority or credibility",
    "Equivocation": "Using ambiguous language or the same term with different meanings",
}

# エージェントタイプの定義を変更
AGENT_TYPES = {"neutral": 50, "pro_misinformation": 50}


class MisinformationGenerator:
    """Misinformation Generation Class"""

    def __init__(self, llm):
        self.llm = llm
        # self.misinformation_descriptions = {
        #     "Satire/parody": "Satire or parody that may mislead without intent to harm",
        #     "Misleading content": "Issues or personal framing due to misuse of information",
        #     "False connection": "Headlines, images, or captions that do not support the content",
        #     "False context": "Genuine content shared with false contextual information",
        #     "Imposter content": "Content impersonating genuine sources",
        #     "Manipulated content": "Content or images manipulated to deceive",
        #     "Fabricated content": "100% false content created to deceive and harm",
        # }
        self.misinformation_descriptions = LOGICAL_FALLACIES

    def generate(self, theme, misinformation_type):
        """Generate misinformation based on the specified theme and type"""
        prompt = MISINFORMATION_GENERATION_PROMPT.format(
            theme=theme,
            misinformation_type=misinformation_type,
            misinformation_description=self.misinformation_descriptions[misinformation_type],
        )

        messages = [{"role": "user", "content": prompt}]
        result = self.llm(messages=messages, temperature=0.7, max_tokens=1000)
        result = result.replace("```json", "").replace("```", "")

        try:
            result_json = json.loads(result)
            return {
                "title": result_json.get("title", ""),
                "content": result_json.get("content", ""),
                "theme": theme,
                "misinformation_type": misinformation_type,
            }
        except json.JSONDecodeError:
            print(f"JSONデコードエラー。テキスト解析を試みます: {result}")
            title_line = [line for line in result.split("\n") if "Title:" in line]
            content_lines = []
            content_started = False

            for line in result.split("\n"):
                if "Content:" in line:
                    content_started = True
                    content_lines.append(line.replace("Content:", "").strip())
                elif content_started:
                    content_lines.append(line.strip())

            title = title_line[0].replace("Title:", "").strip() if title_line else ""
            content = "\n".join(content_lines)

            return {"title": title, "content": content, "theme": theme, "misinformation_type": misinformation_type}


class RebuttalGenerator:
    """Rebuttal Generation Class"""

    def __init__(self, llm):
        self.llm = llm

    def generate(self, misinformation, rebuttal_type):
        """Generate a rebuttal based on the specified misinformation and rebuttal type"""
        prompt = REBUTTAL_GENERATION_PROMPT.format(
            misinformation=f"{misinformation['title']}\n{misinformation['content']}",
            rebuttal_type=rebuttal_type,
            rebuttal_description=REBUTTAL_DESCRIPTIONS[rebuttal_type],
        )

        messages = [{"role": "user", "content": prompt}]
        result = self.llm(messages=messages, temperature=0.7, max_tokens=1000)

        try:
            result_json = json.loads(result)
            return {"content": result_json.get("rebuttal", ""), "rebuttal_type": rebuttal_type, "misinformation_id": misinformation.get("id", "")}
        except json.JSONDecodeError:
            print(f"JSONデコードエラー。テキスト解析を試みます: {result}")
            rebuttal_content = ""
            rebuttal_started = False

            for line in result.split("\n"):
                if "Rebuttal:" in line:
                    rebuttal_started = True
                    rebuttal_content += line.replace("Rebuttal:", "").strip() + "\n"
                elif rebuttal_started:
                    rebuttal_content += line.strip() + "\n"

            return {"content": rebuttal_content.strip(), "rebuttal_type": rebuttal_type, "misinformation_id": misinformation.get("id", "")}


class EvaluationPlanning(PlanningBase):
    """Planning module for evaluation"""

    def __init__(self, llm):
        super().__init__(llm=llm)

    def __call__(self, task_description):
        self.plan = [
            {
                "description": "Get agent information",
                "reasoning instruction": "Retrieve agent information to understand evaluation perspective",
                "tool use instruction": {task_description["agent_type"]},
            },
            {
                "description": "Get misinformation content",
                "reasoning instruction": "Retrieve the content of the misinformation to be evaluated",
                "tool use instruction": {task_description["misinformation"]["id"]},
            },
            {
                "description": "Get rebuttal content",
                "reasoning instruction": "Retrieve the content of the rebuttal to be evaluated",
                "tool use instruction": {task_description["rebuttal"]["id"]},
            },
            {
                "description": "Get agent's past evaluation history",
                "reasoning instruction": "Retrieve how the agent has evaluated similar content in the past",
                "tool use instruction": {task_description["agent_type"]},
            },
            {
                "description": "Reference similar evaluations",
                "reasoning instruction": "Reference the agent's evaluation patterns for similar content",
                "tool use instruction": "Search for similar cases from memory",
            },
            {
                "description": "Determine evaluation",
                "reasoning instruction": "Determine evaluation considering the agent's perspective and past evaluation patterns",
                "tool use instruction": "Use reasoning module",
            },
        ]
        return self.plan


class EvaluationReasoning(ReasoningBase):
    """Reasoning module for evaluation"""

    def __init__(self, profile_type_prompt, memory, llm):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=memory, llm=llm)

    def __call__(self, task_description: str, feedback: str = ""):
        examples, task_description = self.process_task_description(task_description)
        prompt = f"""You are an evaluator of misinformation rebuttals. Please evaluate the rebuttal based on the following information.
        
{task_description}
"""
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(messages=messages, temperature=0.2, max_tokens=1000)
        return reasoning_result


class EvaluationAgent(SimulationAgent):
    """Evaluation Agent"""

    def __init__(self, llm: LLMBase, agent_type="neutral"):
        super().__init__(llm=llm)
        self.memory = MemoryGenerative(llm=self.llm)
        self.planning = EvaluationPlanning(llm=self.llm)
        self.reasoning = EvaluationReasoning(profile_type_prompt="Misinformation Rebuttal Evaluator", memory=self.memory, llm=self.llm)
        self.agent_type = agent_type
        self.past_evaluations = []

    def workflow(self):
        try:
            self.planning(task_description=self.task)
            misinformation = self.task["misinformation"]
            rebuttal = self.task["rebuttal"]

            for evaluation in self.past_evaluations:
                evaluation_text = f"""
                Misinformation Type: {evaluation.get('misinformation_type', '')}
                Rebuttal Type: {evaluation.get('rebuttal_type', '')}
                Persuasiveness: {evaluation.get('persuasiveness', '')}
                Accuracy: {evaluation.get('accuracy', '')}
                Acceptability: {evaluation.get('acceptability', '')}
                Logical Coherence: {evaluation.get('logical_coherence', '')}
                Overall Evaluation: {evaluation.get('overall', '')}
                """
                self.memory(f"past_evaluation: {evaluation_text}")

            similar_evaluations = self.memory(
                f"Misinformation Type: {misinformation['misinformation_type']}, Rebuttal Type: {rebuttal['rebuttal_type']}"
            )

            prompt = EVALUATION_PROMPT.format(
                misinformation=f"{misinformation['title']}\n{misinformation['content']}",
                rebuttal=rebuttal["content"],
                agent_type=self.agent_type,
                past_evaluations=similar_evaluations,
            )

            messages = [{"role": "user", "content": prompt}]
            result = self.llm(messages=messages, temperature=0.2, max_tokens=1000)

            try:
                evaluation = json.loads(result)
                if "rating" not in evaluation:
                    raise ValueError("Rating is not in evaluation")
                if "reasons" not in evaluation or not isinstance(evaluation["reasons"], list):
                    evaluation["reasons"] = []
                if "explanation" not in evaluation:
                    evaluation["explanation"] = ""
            except json.JSONDecodeError:
                print(f"JSONデコードエラー。テキスト解析を試みます: {result}")
                evaluation = {}
                rating = ""
                reasons = []
                explanation = ""

                for line in result.split("\n"):
                    if "Rating:" in line:
                        rating_text = line.replace("Rating:", "").strip().lower()
                        if "helpful" in rating_text:
                            if "not" in rating_text:
                                rating = "not helpful"
                            elif "somewhat" in rating_text:
                                rating = "somewhat helpful"
                            else:
                                rating = "helpful"
                    elif "Reasons:" in line:
                        reasons_text = line.replace("Reasons:", "").strip()
                        reasons = [r.strip() for r in reasons_text.split(",")]
                    elif "Explanation:" in line:
                        explanation = line.replace("Explanation:", "").strip()

                evaluation["rating"] = rating
                evaluation["reasons"] = reasons
                evaluation["explanation"] = explanation

            evaluation["agent_type"] = self.agent_type
            evaluation["agent_id"] = self.task.get("agent_id", "")
            evaluation["misinformation_id"] = misinformation.get("id", "")
            evaluation["rebuttal_id"] = rebuttal.get("id", "")
            evaluation["theme"] = misinformation.get("theme", "")
            evaluation["misinformation_type"] = misinformation.get("misinformation_type", "")
            evaluation["rebuttal_type"] = rebuttal.get("rebuttal_type", "")

            self.past_evaluations.append(evaluation)
            return evaluation

        except Exception as e:
            print(f"Evaluation error: {e}")
            return {
                "error": str(e),
                "agent_type": self.agent_type,
                "misinformation_id": self.task.get("misinformation", {}).get("id", ""),
                "rebuttal_id": self.task.get("rebuttal", {}).get("id", ""),
            }


class MisinformationInteractionTool(CacheInteractionTool):
    """Custom interaction tool for misinformation experiments"""

    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.misinformation_data = []
        self.rebuttal_data = []
        self.evaluation_data = []
        self.agent_evaluation_history = {}

    def load_data(self, misinformations, rebuttals):
        self.misinformation_data = misinformations
        self.rebuttal_data = rebuttals

    def get_misinformation(self, misinformation_id):
        for misinfo in self.misinformation_data:
            if misinfo["id"] == misinformation_id:
                return misinfo
        return None

    def get_rebuttal(self, rebuttal_id):
        for rebuttal in self.rebuttal_data:
            if rebuttal["id"] == rebuttal_id:
                return rebuttal
        return None

    def add_evaluation(self, evaluation):
        self.evaluation_data.append(evaluation)
        agent_type = evaluation.get("agent_type", "")
        agent_id = evaluation.get("agent_id", "")

        if agent_type not in self.agent_evaluation_history:
            self.agent_evaluation_history[agent_type] = []
        self.agent_evaluation_history[agent_type].append(evaluation)

        if not hasattr(self, "agent_id_evaluation_history"):
            self.agent_id_evaluation_history = {}
        if agent_id not in self.agent_id_evaluation_history:
            self.agent_id_evaluation_history[agent_id] = []
        self.agent_id_evaluation_history[agent_id].append(evaluation)

    def get_evaluations(self, agent_type):
        return self.agent_evaluation_history.get(agent_type, [])


def run_experiment():
    """Function to run the experiment with parallel processing improvements"""
    output_dir = "./misinformation_experiment"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize LLM
    llm = OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

    misinformation_generator = MisinformationGenerator(llm)
    rebuttal_generator = RebuttalGenerator(llm)

    # 並列でミスインフォメーション生成
    misinformations = []
    gen_tasks = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for theme in THEMES:
            for misinformation_type in MISINFORMATION_TYPES:
                future = executor.submit(misinformation_generator.generate, theme, misinformation_type)
                gen_tasks.append(future)
        for future in tqdm(concurrent.futures.as_completed(gen_tasks), total=len(gen_tasks), desc="Generating misinformation"):
            misinformation = future.result()
            misinformation["id"] = f"misinfo_{len(misinformations) + 1}"
            misinformations.append(misinformation)

    # 保存
    with open(os.path.join(output_dir, "misinformations.json"), "w", encoding="utf-8") as f:
        json.dump(misinformations, f, ensure_ascii=False, indent=2)

    # 並列でリバタル生成
    rebuttals = []
    rebuttal_tasks = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for misinformation in misinformations:
            for rebuttal_type in REBUTTAL_TYPES:
                future = executor.submit(rebuttal_generator.generate, misinformation, rebuttal_type)
                rebuttal_tasks.append((future, misinformation))
        for future, misinformation in tqdm(rebuttal_tasks, total=len(rebuttal_tasks), desc="Generating rebuttals"):
            rebuttal = future.result()
            rebuttal["id"] = f"rebuttal_{len(rebuttals) + 1}"
            rebuttal["misinformation_id"] = misinformation["id"]
            rebuttals.append(rebuttal)

    with open(os.path.join(output_dir, "rebuttals.json"), "w", encoding="utf-8") as f:
        json.dump(rebuttals, f, ensure_ascii=False, indent=2)

    # Initialize interaction tool
    interaction_tool = MisinformationInteractionTool(output_dir)
    interaction_tool.load_data(misinformations, rebuttals)

    # エージェントの初期化
    agents = {}
    for agent_type, count in AGENT_TYPES.items():
        agents[agent_type] = []
        for i in range(count):
            agent_id = f"{agent_type}_{i+1}"
            agents[agent_type].append({"id": agent_id, "agent": EvaluationAgent(llm, agent_type=agent_type)})

    # 評価タスクの作成
    evaluation_tasks = []
    for misinformation in misinformations:
        related_rebuttals = [r for r in rebuttals if r["misinformation_id"] == misinformation["id"]]
        for rebuttal in related_rebuttals:
            for agent_type, agent_list in agents.items():
                for agent_data in agent_list:
                    task = {
                        "type": "misinformation_evaluation",
                        "misinformation": misinformation,
                        "rebuttal": rebuttal,
                        "agent_type": agent_type,
                        "agent_id": agent_data["id"],
                    }
                    evaluation_tasks.append(task)

    # エージェントごとに評価タスクをグループ化（各エージェントは自身のタスクを順次処理）
    agent_tasks = {}
    for agent_list in agents.values():
        for agent_data in agent_list:
            agent_tasks[agent_data["id"]] = []
    for task in evaluation_tasks:
        agent_tasks[task["agent_id"]].append(task)

    # 各エージェントのタスクを並列処理
    def process_agent_tasks(agent_data, tasks):
        results = []
        for task in tasks:
            agent_data["agent"].task = task
            evaluation = agent_data["agent"].workflow()
            evaluation["agent_id"] = agent_data["id"]
            results.append(evaluation)
            interaction_tool.add_evaluation(evaluation)
        return results

    evaluations = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_agent = {}
        for agent_type, agent_list in agents.items():
            for agent_data in agent_list:
                future = executor.submit(process_agent_tasks, agent_data, agent_tasks[agent_data["id"]])
                future_to_agent[future] = agent_data["id"]
        for future in tqdm(concurrent.futures.as_completed(future_to_agent), total=len(future_to_agent), desc="Executing evaluations"):
            evaluations.extend(future.result())

    with open(os.path.join(output_dir, "evaluations.json"), "w", encoding="utf-8") as f:
        json.dump(evaluations, f, ensure_ascii=False, indent=2)

    return evaluations


if __name__ == "__main__":
    results = run_experiment()
    print("Experiment completed. Results are saved in the ./misinformation_experiment directory.")
