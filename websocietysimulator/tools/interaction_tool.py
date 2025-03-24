import json
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("websocietysimulator")


class InteractionTool:
    def __init__(self, data_dir: str):
        """
        Initialize the tool with the dataset directory.
        Args:
            data_dir: Path to the directory containing Yelp dataset files.
        """
        logger.info(f"Initializing InteractionTool with data directory: {data_dir}")
        self.data_dir = data_dir
        # Convert DataFrames to dictionaries for O(1) lookup
        logger.info(f"Loading item data from {os.path.join(data_dir, 'item.json')}")
        self.item_data = {item["item_id"]: item for item in self._load_data("item.json")}
        logger.info(f"Loading user data from {os.path.join(data_dir, 'user.json')}")
        self.user_data = {user["user_id"]: user for user in self._load_data("user.json")}

        # Create review indices
        logger.info(f"Loading review data from {os.path.join(data_dir, 'review.json')}")
        reviews = self._load_data("review.json")
        self.review_data = {review["review_id"]: review for review in reviews}
        self.item_reviews = {}
        self.user_reviews = {}

        # Build review indices
        logger.info("Building review indices")
        for review in reviews:
            # Index by item_id
            self.item_reviews.setdefault(review["item_id"], []).append(review)
            # Index by user_id
            self.user_reviews.setdefault(review["user_id"], []).append(review)

    def _load_data(self, filename: str) -> List[Dict]:
        """Load data as a list of dictionaries."""
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            return [json.loads(line) for line in file]

    def get_user(self, user_id: str) -> Dict:
        """
        Get user information by user ID.
        Args:
            user_id: User ID.
        Returns:
            User information.
        """
        if hasattr(self, "user_db"):
            # 既存のユーザーDBがある場合はそれを使用
            return self.user_db.get(user_id, {})
        else:
            # コミュニティノート評価用のユーザー情報を取得
            user_file = os.path.join(self.data_dir, "user.json")
            if os.path.exists(user_file):
                with open(user_file, "r", encoding="utf-8") as f:
                    users = json.load(f)
                    for user in users:
                        if user.get("user_id") == user_id:
                            return user
            return {"user_id": user_id, "profile": "No profile information available."}

    def get_item(self, item_id: str = None) -> Optional[Dict]:
        """Fetch item data based on item_id."""
        return self.item_data.get(item_id) if item_id else None

    def get_reviews(self, item_id: Optional[str] = None, user_id: Optional[str] = None, review_id: Optional[str] = None) -> List[Dict]:
        """Fetch reviews filtered by various parameters."""
        if review_id:
            return [self.review_data[review_id]] if review_id in self.review_data else []

        if item_id:
            return self.item_reviews.get(item_id, [])
        elif user_id:
            return self.user_reviews.get(user_id, [])

        return []

    def get_note(self, note_id: str) -> Dict:
        """
        Get community note information by note ID.
        Args:
            note_id: Note ID.
        Returns:
            Community note information.
        """
        note_file = os.path.join(self.data_dir, "note.json")
        if os.path.exists(note_file):
            with open(note_file, "r", encoding="utf-8") as f:
                notes = json.load(f)
                for note in notes:
                    if str(note.get("note_id")) == str(note_id):
                        return note
        return {"note_id": note_id, "content": "No content available.", "sources": []}

    def get_evaluations(self, user_id: str) -> List[Dict]:
        """
        Get past evaluations by user ID.
        Args:
            user_id: User ID.
        Returns:
            List of past evaluations.
        """
        evaluation_file = os.path.join(self.data_dir, "evaluation.json")
        if os.path.exists(evaluation_file):
            with open(evaluation_file, "r", encoding="utf-8") as f:
                evaluations = json.load(f)
                return [eval_data for eval_data in evaluations if eval_data.get("user_id") == user_id]
        return []
