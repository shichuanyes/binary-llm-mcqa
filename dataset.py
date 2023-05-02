from abc import ABC, abstractmethod
from typing import List

from datasets import load_dataset
from pandas import DataFrame

from question import Question


class Dataset(ABC):
    def __init__(self, *args, **kwargs):
        ds = load_dataset(*args, **kwargs)
        shuffled = ds.shuffle(seed=42)
        self.df = DataFrame(shuffled[:len(shuffled) // 10])

    def to_questions(self) -> List[Question]:
        return self.df.apply(self.to_question, axis=1).tolist()

    @staticmethod
    @abstractmethod
    def to_question(row) -> Question:
        pass

    @staticmethod
    @abstractmethod
    def to_answer(answer_idx: int):
        pass
