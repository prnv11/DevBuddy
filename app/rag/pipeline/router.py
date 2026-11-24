"""
Query router used by the dashboard.

Previously this module returned three hard-coded answers after a fake delay.
It now calls the real retrieval pipeline in `ir/` (BM25F + dense hybrid,
cross-framework concept alignment, Rocchio feedback) and composes a cited
answer; see ir/answer.py.
"""
from __future__ import annotations

import os

from ir.answer import Answerer

DATA_DIR = os.getenv("DEVBUDDY_DATA", "data")


class DevBuddyRouter:
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir

    @property
    def answerer(self) -> Answerer:
        return Answerer.get(self.data_dir)          # loaded once, on first use

    def invoke(self, input_data: dict) -> dict:
        return self.answerer.answer(input_data["question"], input_data.get("primary"), input_data.get("learning"))

    def refine(self, input_data: dict) -> dict:
        return self.answerer.refine(input_data["question"], input_data.get("relevant", []),
                                    input_data.get("non_relevant", []), input_data.get("primary"),
                                    input_data.get("learning"))


def build_router() -> DevBuddyRouter:
    return DevBuddyRouter()
