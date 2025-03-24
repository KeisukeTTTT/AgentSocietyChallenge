import json
import logging
import os
from typing import Dict, Iterator, List, Optional

import lmdb
from tqdm import tqdm

logger = logging.getLogger("websocietysimulator")


class CacheInteractionTool:
    def __init__(self, data_dir: str):
        """
        Initialize the tool with the dataset directory.
        Args:
            data_dir: Path to the directory containing Yelp dataset files.
        """
        logger.info(f"Initializing InteractionTool with data directory: {data_dir}")
        self.data_dir = data_dir

        # Create LMDB environments
        self.env_dir = os.path.join(data_dir, "lmdb_cache")
        os.makedirs(self.env_dir, exist_ok=True)

        self.user_env = lmdb.open(os.path.join(self.env_dir, "users"), map_size=2 * 1024 * 1024 * 1024)
        self.item_env = lmdb.open(os.path.join(self.env_dir, "items"), map_size=2 * 1024 * 1024 * 1024)
        self.review_env = lmdb.open(os.path.join(self.env_dir, "reviews"), map_size=8 * 1024 * 1024 * 1024)

        # Initialize the database if empty
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the database with data from files."""
        # ユーザーデータの読み込み
        self.user_db = {}
        for user in tqdm(self._iter_file("user.json"), desc="Loading users"):
            self.user_db[user["user_id"]] = user

        # コミュニティノートデータの読み込み
        self.note_db = {}
        note_file = os.path.join(self.data_dir, "note.json")
        if os.path.exists(note_file):
            for note in tqdm(self._iter_file("note.json"), desc="Loading notes"):
                self.note_db[note["note_id"]] = note

        # 評価データの読み込み
        self.evaluation_db = {}
        evaluation_file = os.path.join(self.data_dir, "evaluation.json")
        if os.path.exists(evaluation_file):
            for evaluation in tqdm(self._iter_file("evaluation.json"), desc="Loading evaluations"):
                user_id = evaluation["user_id"]
                if user_id not in self.evaluation_db:
                    self.evaluation_db[user_id] = []
                self.evaluation_db[user_id].append(evaluation)

        # Yelp形式のデータがある場合は読み込む
        business_file = os.path.join(self.data_dir, "business.json")
        if os.path.exists(business_file):
            self.business_db = {}
            for business in tqdm(self._iter_file("business.json"), desc="Loading businesses"):
                self.business_db[business["business_id"]] = business

            # レビューデータの読み込み
            self.review_db = {}
            for review in tqdm(self._iter_file("review.json"), desc="Loading reviews"):
                business_id = review.get("business_id")
                if business_id:
                    if business_id not in self.review_db:
                        self.review_db[business_id] = []
                    self.review_db[business_id].append(review)

            # ユーザーレビュー履歴の構築
            self.user_review_history = {}
            for review in tqdm(self._iter_file("review.json"), desc="Building user review history"):
                user_id = review.get("user_id")
                if user_id:
                    if user_id not in self.user_review_history:
                        self.user_review_history[user_id] = []
                    self.user_review_history[user_id].append(review)

    def _iter_file(self, filename: str) -> Iterator[Dict]:
        """Iterate through file line by line."""

        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} does not exist.")
            return

        try:
            # まず配列全体のJSONとして読み込みを試みる（コミュニティノート形式）
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list):
                    for item in data:
                        yield item
                    return
        except json.JSONDecodeError:
            # 配列全体のJSONとして読み込めない場合は、一行ごとのJSONとして読み込む（Yelp形式）
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if line:  # 空行をスキップ
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing JSON in {filename}: {e}")
                            continue

    def get_user(self, user_id: str) -> Dict:
        """
        Get user information by user ID.
        Args:
            user_id: User ID.
        Returns:
            User information.
        """
        return self.user_db.get(user_id, {"user_id": user_id, "profile": "No profile information available."})

    def get_note(self, note_id: str) -> Dict:
        """
        Get community note information by note ID.
        Args:
            note_id: Note ID.
        Returns:
            Community note information.
        """
        return self.note_db.get(note_id, {"note_id": note_id, "content": "No content available.", "sources": []})

    def get_evaluations(self, user_id: str) -> List[Dict]:
        """
        Get past evaluations by user ID.
        Args:
            user_id: User ID.
        Returns:
            List of past evaluations.
        """
        return self.evaluation_db.get(user_id, [])

    def get_item(self, item_id: str) -> Optional[Dict]:
        """Fetch item data based on item_id."""
        if not item_id:
            return None

        with self.item_env.begin() as txn:
            item_data = txn.get(item_id.encode())
            if item_data:
                return json.loads(item_data)
        return None

    def get_reviews(self, item_id: Optional[str] = None, user_id: Optional[str] = None, review_id: Optional[str] = None) -> List[Dict]:
        """Fetch reviews filtered by various parameters."""
        if review_id:
            with self.review_env.begin() as txn:
                review_data = txn.get(review_id.encode())
                if review_data:
                    return [json.loads(review_data)]
            return []

        with self.review_env.begin() as txn:
            if item_id:
                review_ids = json.loads(txn.get(f"item_{item_id}".encode()) or "[]")
            elif user_id:
                review_ids = json.loads(txn.get(f"user_{user_id}".encode()) or "[]")
            else:
                return []

            # Fetch complete review data for each review_id
            reviews = []
            for rid in review_ids:
                review_data = txn.get(rid.encode())
                if review_data:
                    reviews.append(json.loads(review_data))
            return reviews

    def __del__(self):
        """Cleanup LMDB environments on object destruction."""
        self.user_env.close()
        self.item_env.close()
        self.review_env.close()
