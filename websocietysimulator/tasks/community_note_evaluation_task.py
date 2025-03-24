from typing import Any, Dict


class CommunityNoteEvaluationTask:
    def __init__(self, user_id: str, note_id: str):
        """
        Community Note Evaluation Task for the CommunityNoteEvaluationAgent.
        Args:
            user_id: The ID of the user evaluating the community note.
            note_id: The ID of the community note to be evaluated.
        """
        self.user_id = user_id
        self.note_id = note_id

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task to a dictionary.
        Returns:
            dict: The task in dictionary format.
        """
        return {
            "description": """This is a community note evaluation task. 
            You are an evaluation agent that evaluates community notes. 
            There is a user with id and a community note with id.""",
            "user_id": self.user_id,
            "note_id": self.note_id,
        }
