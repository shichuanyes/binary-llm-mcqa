from abc import ABC, abstractmethod
from typing import List, Dict

from question import Question


class Method(ABC):
    @staticmethod
    @abstractmethod
    def get_prompts(question: Question) -> List[str]:
        pass

    @staticmethod
    @abstractmethod
    def get_completions(question: Question) -> List[str]:
        pass

    @classmethod
    def prepare_finetune(cls, question: Question) -> List[Dict[str, str]]:
        result = []
        for prompt, completion in zip(cls.get_prompts(question), cls.get_completions(question)):
            result.append(
                {
                    "prompt": prompt,
                    "completion": completion
                }
            )
        return result

    @classmethod
    @abstractmethod
    def ask(cls, question: Question, model: str) -> int:
        pass

    @staticmethod
    def check_answer(question: Question, answer_idx: int) -> bool:
        return question.answer_idx == answer_idx
