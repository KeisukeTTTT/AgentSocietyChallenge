import json
import logging
import os

from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.agent.modules.memory_modules import MemoryGenerative
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.llm import LLMBase, OpenAILLM

logging.basicConfig(level=logging.INFO)


# 評価指示と基準を共通化するための定数
EVALUATION_INSTRUCTIONS = """
Please consider the following points for your evaluation:
1. Is it consistent with your past evaluation patterns?
2. Is the community note content accurate and helpful?
3. Is the community note written from a neutral perspective?
4. Does the community note present sufficient evidence?

Your evaluation must be one of three ratings:
- helpful: The note is accurate, well-sourced, and adds valuable context
- somewhat helpful: The note has some value but also has issues or limitations
- not helpful: The note is inaccurate, misleading, or otherwise problematic

For "helpful" ratings, select one or more reasons from:
- Cites high-quality sources
- Easy to understand
- Directly addresses the post's claim
- Provides important context
- Neutral or unbiased language
- Other

For "not helpful" ratings, select one or more reasons from:
- Sources not included or unreliable
- Sources do not support note
- Incorrect information
- Opinion or speculation
- Typos or unclear language
- Misses key points or irrelevant
- Argumentative or biased language
- Note not needed on this post
- Other

For "somewhat helpful" ratings, you may select reasons from either or both of the above lists.

Please output your evaluation in the following format:
Evaluation: [helpful/somewhat helpful/not helpful]
Reasons: [comma-separated list of reasons]
Explanation: [brief explanation of your evaluation]
"""


class CommunityNotePlanning(PlanningBase):
    """Planning module for community note evaluation"""

    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)

    def __call__(self, task_description):
        """Override the parent class's __call__ method"""
        self.plan = [
            {
                "description": "Get user information",
                "reasoning instruction": "Retrieve user information to understand past evaluation patterns",
                "tool use instruction": {task_description["user_id"]},
            },
            {
                "description": "Get community note content",
                "reasoning instruction": "Retrieve the content of the community note to be evaluated",
                "tool use instruction": {task_description["note_id"]},
            },
            {
                "description": "Get user's past evaluation history",
                "reasoning instruction": "Retrieve how the user has evaluated community notes in the past",
                "tool use instruction": {task_description["user_id"]},
            },
            {
                "description": "Reference similar community note evaluations",
                "reasoning instruction": "Reference the user's evaluation patterns for similar community notes",
                "tool use instruction": "Search for similar cases from memory",
            },
            {
                "description": "Determine evaluation",
                "reasoning instruction": "Determine evaluation considering the user's past evaluation patterns and current community note content",
                "tool use instruction": "Use reasoning module",
            },
        ]
        return self.plan


class CommunityNoteReasoning(ReasoningBase):
    """Reasoning module for community note evaluation"""

    def __init__(self, profile_type_prompt, memory, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=memory, llm=llm)

    def __call__(self, task_description: str, feedback: str = ""):
        """Override the parent class's __call__ method"""
        examples, task_description = self.process_task_description(task_description)

        prompt = f"""You are a community note evaluator. Please evaluate the community note based on the following information.
        
{task_description}

{EVALUATION_INSTRUCTIONS}
"""

        with open("./prompt.txt", "w") as f:
            f.write(prompt)

        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(messages=messages, temperature=0.1, max_tokens=1000)

        return reasoning_result


class CommunityNoteEvaluationAgent(SimulationAgent):
    """Community Note Evaluation Agent"""

    def __init__(self, llm: LLMBase):
        """Initialize the Community Note Evaluation Agent"""
        super().__init__(llm=llm)
        self.memory = MemoryGenerative(llm=self.llm)
        self.planning = CommunityNotePlanning(llm=self.llm)
        self.reasoning = CommunityNoteReasoning(profile_type_prompt="Community Note Evaluator", memory=self.memory, llm=self.llm)

    def workflow(self):
        """
        Workflow for community note evaluation
        Returns:
            dict: {"evaluation": str, "reasons": list, "explanation": str}
        """
        try:
            # Execute plan
            self.planning(task_description=self.task)

            # Get user information
            # user = str(self.interaction_tool.get_user(user_id=self.task["user_id"]))

            # Get community note content
            community_note = self.interaction_tool.get_note(note_id=self.task["note_id"])
            note_content = community_note["content"]

            # Get user's past evaluation history
            all_evaluations = self.interaction_tool.get_evaluations(user_id=self.task["user_id"])

            # Filter out the current note from past evaluations
            past_evaluations = [eval for eval in all_evaluations if eval["note_id"] != self.task["note_id"]]

            # Add past evaluations to memory
            for evaluation in past_evaluations:
                note = self.interaction_tool.get_note(note_id=evaluation["note_id"])
                evaluation_text = f"""
                Note content: {note["content"]}
                Evaluation: {evaluation['evaluation']}
                Reasons: {evaluation.get('reasons', '')}
                Explanation: {evaluation.get('explanation', '')}
                """
                self.memory(f"review: {evaluation_text}")

            # Reference similar community note evaluations
            similar_evaluation = self.memory(f"{community_note}")

            # Prepare information for determining evaluation
            task_description = f"""
Community note to evaluate: {note_content}

Your past evaluations of similar community notes: {similar_evaluation}

"""

            # Execute reasoning
            result = self.reasoning(task_description)

            try:
                # Parse results
                evaluation_line = [line for line in result.split("\n") if "Evaluation:" in line][0]
                reasons_line = [line for line in result.split("\n") if "Reasons:" in line][0]
                explanation_line = [line for line in result.split("\n") if "Explanation:" in line][0]

                evaluation = evaluation_line.split(":")[1].strip()
                reasons_text = reasons_line.split(":")[1].strip()
                explanation = explanation_line.split(":")[1].strip()

                # Convert reasons to list
                reasons = [reason.strip() for reason in reasons_text.split(",")]

                # Ensure evaluation is one of the three valid options
                valid_evaluations = ["helpful", "somewhat helpful", "not helpful"]
                if evaluation not in valid_evaluations:
                    # Try to match to closest valid evaluation
                    if "helpful" in evaluation.lower() and "not" in evaluation.lower():
                        evaluation = "not helpful"
                    elif "somewhat" in evaluation.lower() or "some" in evaluation.lower():
                        evaluation = "somewhat helpful"
                    elif "helpful" in evaluation.lower():
                        evaluation = "helpful"
                    else:
                        evaluation = "not helpful"

                # Validate reasons based on evaluation
                helpful_reasons = [
                    "Cites high-quality sources",
                    "Easy to understand",
                    "Directly addresses the post's claim",
                    "Provides important context",
                    "Neutral or unbiased language",
                    "Other",
                ]

                not_helpful_reasons = [
                    "Sources not included or unreliable",
                    "Sources do not support note",
                    "Incorrect information",
                    "Opinion or speculation",
                    "Typos or unclear language",
                    "Misses key points or irrelevant",
                    "Argumentative or biased language",
                    "Note not needed on this post",
                    "Other",
                ]

                # For somewhat helpful, any reason is valid
                if evaluation == "helpful":
                    # Filter to only include valid helpful reasons
                    valid_reasons = []
                    for reason in reasons:
                        matched = False
                        for valid_reason in helpful_reasons:
                            if valid_reason.lower() in reason.lower():
                                valid_reasons.append(valid_reason)
                                matched = True
                                break
                        if not matched and reason.strip():
                            valid_reasons.append("Other")
                    reasons = valid_reasons if valid_reasons else ["Other"]

                elif evaluation == "not helpful":
                    # Filter to only include valid not helpful reasons
                    valid_reasons = []
                    for reason in reasons:
                        matched = False
                        for valid_reason in not_helpful_reasons:
                            if valid_reason.lower() in reason.lower():
                                valid_reasons.append(valid_reason)
                                matched = True
                                break
                        if not matched and reason.strip():
                            valid_reasons.append("Other")
                    reasons = valid_reasons if valid_reasons else ["Other"]

                # For somewhat helpful, any reason is valid, but normalize them
                elif evaluation == "somewhat helpful":
                    valid_reasons = []
                    all_valid_reasons = helpful_reasons + not_helpful_reasons
                    for reason in reasons:
                        matched = False
                        for valid_reason in all_valid_reasons:
                            if valid_reason.lower() in reason.lower():
                                valid_reasons.append(valid_reason)
                                matched = True
                                break
                        if not matched and reason.strip():
                            valid_reasons.append("Other")
                    reasons = valid_reasons if valid_reasons else ["Other"]

                return {"evaluation": evaluation, "reasons": reasons, "explanation": explanation}
            except Exception as e:
                print(f"Result parsing error: {e}")
                print(f"Result: {result}")
                return {"evaluation": "not helpful", "reasons": ["Other"], "explanation": "Could not determine evaluation."}

        except Exception as e:
            print(f"Workflow error: {e}")
            return {"evaluation": "not helpful", "reasons": ["Other"], "explanation": "An error occurred."}


if __name__ == "__main__":
    # Check data directory
    data_dir = "./community_notes"

    # Check if training_evaluation.json exists
    training_file_exists = os.path.exists(os.path.join(data_dir, "training_evaluation.json"))

    # If training file exists, create a custom interaction tool that uses it
    if training_file_exists:
        from websocietysimulator.tools import CacheInteractionTool

        class TrainingInteractionTool(CacheInteractionTool):
            def __init__(self, data_dir):
                super().__init__(data_dir)
                # Load training evaluation data instead of all evaluation data
                with open(os.path.join(data_dir, "training_evaluation.json"), "r") as f:
                    self.evaluation_data = json.load(f)

        # Use the custom interaction tool
        interaction_tool = TrainingInteractionTool(data_dir)
        simulator = Simulator(device="gpu")
        simulator.set_interaction_tool(interaction_tool)
    else:
        # Use the standard simulator setup
        simulator = Simulator(data_dir=data_dir, device="gpu", cache=True)

    simulator.set_task_and_groundtruth(task_dir=os.path.join(data_dir, "tasks"), groundtruth_dir=os.path.join(data_dir, "groundtruth"))

    # Setup agent and LLM
    simulator.set_agent(CommunityNoteEvaluationAgent)
    simulator.set_llm(OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini"))

    # Run simulation
    outputs = simulator.run_simulation(number_of_tasks=10, enable_threading=True, max_workers=4)

    # Evaluate agent
    evaluation_results = simulator.evaluate()
    with open("./evaluation_results_community_notes.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)

    # Get evaluation history
    evaluation_history = simulator.get_evaluation_history()
    with open("./evaluation_history_community_notes.json", "w") as f:
        json.dump(evaluation_history, f, indent=4)

    # 詳細な出力とgroundtruthを保存
    detailed_outputs = {}
    for i, output in enumerate(outputs):
        if output is not None:
            task_id = f"task_{i+1}"
            task = simulator.tasks[i].to_dict() if i < len(simulator.tasks) else {}
            agent_output = output.get("output", {})
            groundtruth = simulator.groundtruth_data[i] if i < len(simulator.groundtruth_data) else {}
            detailed_outputs[task_id] = {"task": task, "agent_output": agent_output, "groundtruth": groundtruth}

    with open("./detailed_outputs_community_notes.json", "w") as f:
        json.dump(detailed_outputs, f, indent=4)
