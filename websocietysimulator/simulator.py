import json
import logging
import os
from typing import Any, Dict, List, Type, Union

from .agent.recommendation_agent import RecommendationAgent
from .agent.simulation_agent import SimulationAgent
from .llm import LLMBase
from .tasks.community_note_evaluation_task import CommunityNoteEvaluationTask
from .tasks.recommendation_task import RecommendationTask
from .tasks.simulation_task import SimulationTask
from .tools import CacheInteractionTool, InteractionTool
from .tools.evaluation_tool import CommunityNoteEvaluator, RecommendationEvaluator, SimulationEvaluator

logger = logging.getLogger("websocietysimulator")


class Simulator:

    def __init__(self, data_dir: str = None, device: str = "auto", cache: bool = False):
        """
        Initialize the Simulator.
        Args:
            data_dir: Path to the directory containing Yelp dataset files.
            device: Device to use for evaluation. "auto" (default) will use GPU if available, otherwise CPU. Available options: "gpu", "cpu", "auto".
            cache: Whether to use cache for interaction tool.
        """
        logger.info("Start initializing Simulator")
        self.data_dir = data_dir
        if data_dir is None:
            self.interaction_tool = None
        else:
            if cache:
                logger.info("Using CacheInteractionTool")
                self.interaction_tool = CacheInteractionTool(data_dir)
            else:
                logger.info("Using Normal InteractionTool")
                self.interaction_tool = InteractionTool(data_dir)

        self.tasks = []  # List to store tasks
        self.groundtruth_data = []  # List to store groundtruth data
        self.agent_class = None
        self.llm = None
        self.recommendation_evaluator = RecommendationEvaluator()
        self.simulation_evaluator = SimulationEvaluator(device)
        self.community_note_evaluator = CommunityNoteEvaluator()
        self.simulation_outputs = []
        self.evaluation_results = []
        logger.info("Simulator initialized")

    def set_interaction_tool(self, interaction_tool: Union[InteractionTool, CacheInteractionTool]):
        self.interaction_tool = interaction_tool

    def set_task_and_groundtruth(self, task_dir: str, groundtruth_dir: str):
        """
        Load tasks from a directory.
        Args:
            task_dir: Directory containing task files.
            groundtruth_dir: Directory containing groundtruth files.
        """
        self.tasks = []  # Clear previous tasks
        self.groundtruth_data = []

        # Get all task files and sort them by index
        task_files = sorted(
            [f for f in os.listdir(task_dir) if f.startswith("task_") and f.endswith(".json")], key=lambda x: int(x.split("_")[1].split(".")[0])
        )

        for task_file in task_files:
            # Get the corresponding groundtruth file
            task_index = task_file.split("_")[1].split(".")[0]
            groundtruth_file = f"groundtruth_{task_index}.json"
            groundtruth_path = os.path.join(groundtruth_dir, groundtruth_file)

            if not os.path.exists(groundtruth_path):
                logger.warning(f"Groundtruth file {groundtruth_file} not found for task {task_file}")
                continue

            # Read task file
            task_path = os.path.join(task_dir, task_file)
            with open(task_path, "r") as f:
                task_data = json.load(f)
                task_type = task_data.get("type")

                # Determine scenario type and create corresponding object
                if task_type == "user_behavior_simulation":
                    task = SimulationTask(user_id=task_data["user_id"], item_id=task_data["item_id"])
                elif task_type == "recommendation":
                    task = RecommendationTask(
                        user_id=task_data["user_id"],
                        candidate_category=task_data["candidate_category"],
                        candidate_list=task_data["candidate_list"],
                        loc=task_data["loc"],
                    )
                elif task_type == "community_note_evaluation":
                    task = CommunityNoteEvaluationTask(
                        user_id=task_data["user_id"],
                        note_id=task_data["note_id"],
                    )
                else:
                    raise ValueError(f"Unsupported task type: {task_type}")

            with open(groundtruth_path, "r") as f:
                groundtruth_data = json.load(f)

            self.tasks.append(task)
            self.groundtruth_data.append(groundtruth_data)

        logger.info(f"Loaded {len(self.tasks)} task-groundtruth pairs")

    def set_agent(self, agent_class: Type):
        """
        Set the agent class to be used for the simulation.
        Args:
            agent_class: A class inheriting from the abstract Agent class.
        """
        if not issubclass(agent_class, (SimulationAgent, RecommendationAgent)):
            raise ValueError("Agent class must inherit from SimulationAgent or RecommendationAgent.")
        self.agent_class = agent_class
        logger.info("Agent class set")

    def set_llm(self, llm: Union[LLMBase, list[LLMBase]]):
        """
        Set the LLM to be used for the simulation.
        Args:
            llm: A class inheriting from the abstract LLM class.
        """
        self.llm = llm
        logger.info("LLM set")

    def run_simulation(
        self, number_of_tasks: int = None, enable_threading: bool = False, max_workers: int = None, time_limitation: float = None
    ) -> List[Any]:
        """
        Run the simulation with optional multi-threading support and time limitation.

        Args:
            number_of_tasks: Number of tasks to run. If None, run all tasks.
            enable_threading: Whether to enable multi-threading. Default is False.
            max_workers: Maximum number of threads to use. If None, will use min(32, number_of_tasks).
            time_limitation: Time limit in minutes. If None, no time limit is applied.
        Returns:
            List of outputs from agents for each scenario.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

        start_time = time.time()
        timeout_seconds = time_limitation * 60 if time_limitation else None

        logger.info("Running simulation")
        if not self.agent_class:
            raise RuntimeError("Agent class is not set. Use set_agent() to set it.")
        if not self.interaction_tool:
            raise RuntimeError("Interaction tool is not set. Use set_interaction_tool() to set it.")

        task_to_run = self.tasks[:number_of_tasks] if number_of_tasks is not None else self.tasks
        logger.info(f"Total tasks: {len(task_to_run)}")

        # If multi-threading is not enabled, use original serial processing
        if not enable_threading:
            self.simulation_outputs = []
            for index, task in enumerate(task_to_run):
                # Check if timeout has occurred
                if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                    logger.warning(f"Time limit ({time_limitation} minutes) reached. Stopping simulation.")
                    break

                if isinstance(self.llm, list):
                    agent = self.agent_class(llm=self.llm[index % len(self.llm)])
                else:
                    agent = self.agent_class(llm=self.llm)
                agent.set_interaction_tool(self.interaction_tool)
                agent.insert_task(task)

                try:
                    output = agent.workflow()
                    result = {"task": task.to_dict(), "output": output}
                except NotImplementedError:
                    result = {"task": task.to_dict(), "error": "Forward method not implemented by participant."}
                self.simulation_outputs.append(result)
                logger.info(f"Simulation finished for task {index}")
        else:
            # Multi-threading processing
            from threading import Event, Lock

            log_lock = Lock()
            cancel_event = Event()  # Add cancellation event flag
            self.simulation_outputs = [None] * len(task_to_run)

            def process_task(task_index_tuple):
                from concurrent.futures import ThreadPoolExecutor, TimeoutError

                def run_agent_task(agent, task):
                    output = agent.workflow()
                    return output

                index, task = task_index_tuple
                # Check if cancellation has been requested
                if cancel_event.is_set():
                    return index, None

                if isinstance(self.llm, list):
                    agent = self.agent_class(llm=self.llm[index % len(self.llm)])
                else:
                    agent = self.agent_class(llm=self.llm)
                agent.set_interaction_tool(self.interaction_tool)
                agent.insert_task(task)

                try:
                    # Use internal ThreadPoolExecutor to execute a single task with a 5-minute timeout
                    with ThreadPoolExecutor(max_workers=1) as single_task_executor:
                        future = single_task_executor.submit(run_agent_task, agent, task)
                        try:
                            output = future.result(timeout=300)  # 5 minutes timeout
                            result = {"task": task.to_dict(), "output": output}
                        except TimeoutError:
                            logger.warning(f"Task {index} timed out")
                            # Force close executor
                            single_task_executor._threads.clear()
                            single_task_executor.shutdown(wait=False)
                            return index, None
                except NotImplementedError:
                    result = {"task": task.to_dict(), "error": "Forward method not implemented by participant."}
                except Exception as e:
                    logger.error(f"Task {index} failed with error: {str(e)}")
                    return index, None

                with log_lock:
                    logger.info(f"Simulation finished for task {index}")

                return index, result

            # Determine number of threads
            if max_workers is None:
                max_workers = min(32, len(task_to_run))
            else:
                max_workers = min(max_workers, len(task_to_run))

            logger.info(f"Running with {max_workers} threads")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {executor.submit(process_task, (i, task)): i for i, task in enumerate(task_to_run)}

                try:
                    for future in as_completed(future_to_index, timeout=timeout_seconds):
                        try:
                            index, result = future.result()
                            self.simulation_outputs[index] = result
                        except Exception as e:
                            logger.error(f"Task failed with error: {str(e)}")
                except TimeoutError:
                    logger.error(f"Time limit ({time_limitation} minutes) reached.")
                    # Set cancellation flag
                    cancel_event.set()
                    # Force cancel all tasks
                    for future in future_to_index:
                        future.cancel()
                    # Immediately shut down the executor without waiting for tasks to complete
                    executor._threads.clear()
                    executor.shutdown(wait=False)
                    raise TimeoutError

        logger.info("Simulation finished")
        # Filter out None values (incomplete tasks)
        return self.simulation_outputs

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the simulation results using the loaded groundtruth data.
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating simulation results")
        if not self.simulation_outputs:
            raise RuntimeError("No simulation outputs to evaluate. Run simulation first.")

        # Check the number of data entries
        sim_count = len(self.simulation_outputs)
        gt_count = len(self.groundtruth_data)

        if sim_count != gt_count:
            logger.warning(f"Warning: Number of simulation outputs ({sim_count}) does not match ground truth data ({gt_count})")
            # Use the smaller number
            eval_count = min(sim_count, gt_count)
            groundtruth_data = self.groundtruth_data[:eval_count]
            self.simulation_outputs = self.simulation_outputs[:eval_count]
        else:
            groundtruth_data = self.groundtruth_data

        evaluation_results = {}

        # Choose evaluation method based on agent type
        if issubclass(self.agent_class, RecommendationAgent):
            evaluation_results = self._evaluate_recommendation(groundtruth_data)
        elif issubclass(self.agent_class, SimulationAgent):
            # Check task type
            if self.tasks and hasattr(self.tasks[0], "__class__") and self.tasks[0].__class__.__name__ == "CommunityNoteEvaluationTask":
                evaluation_results = self._evaluate_community_note(groundtruth_data)
            else:
                evaluation_results = self._evaluate_simulation(groundtruth_data)

        # Add data entry information to evaluation results
        evaluation_results["data_info"] = {
            "evaluated_count": eval_count if sim_count != gt_count else sim_count,
            "original_simulation_count": sim_count,
            "original_ground_truth_count": gt_count,
        }

        self.evaluation_results.append(evaluation_results)
        logger.info("Evaluation finished")
        return evaluation_results

    def _evaluate_recommendation(self, ground_truth_data: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate recommendation results using groundtruth
        """
        # Extract real POIs from ground truth data
        gt_pois = [item["ground truth"] for item in ground_truth_data]

        pred_pois = []
        for output in self.simulation_outputs:
            if output is not None:
                pred_pois.append(output["output"])
            else:
                pred_pois.append([""])

        # Calculate evaluation metrics
        metrics = self.recommendation_evaluator.calculate_hr_at_n(
            ground_truth=gt_pois,
            predictions=pred_pois,
        )

        return {
            "type": "recommendation",
            "metrics": metrics.__dict__,
        }

    def _evaluate_simulation(self, ground_truth_data: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate simulation results
        """
        simulated_data = []
        for output in self.simulation_outputs:
            if output is not None:
                simulated_data.append(output["output"])
            else:
                simulated_data.append({"stars": 0, "review": ""})
        metrics = self.simulation_evaluator.calculate_metrics(simulated_data=simulated_data, real_data=ground_truth_data)
        return {
            "type": "simulation",
            "metrics": metrics.__dict__,
        }

    def _evaluate_community_note(self, ground_truth_data: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate community note evaluation results
        """
        simulated_data = []
        for output in self.simulation_outputs:
            if output is not None:
                simulated_data.append(output["output"])
            else:
                simulated_data.append({"evaluation": "", "reasons": [""], "explanation": ""})

        metrics = self.community_note_evaluator.calculate_metrics(simulated_data=simulated_data, real_data=ground_truth_data)

        return {
            "type": "community_note_evaluation",
            "metrics": metrics.__dict__,
        }

    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of evaluation results
        Returns:
            List of evaluation results
        """
        return self.evaluation_results
