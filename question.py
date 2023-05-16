from dataclasses import dataclass
from typing import List

from utils import idx_to_letter


@dataclass
class Part:
    text: str
    tag: str = None

    def __str__(self):
        if self.tag is None:
            return self.text
        return f"{self.tag}: {self.text}"


@dataclass
class Question:
    parts: List[Part]
    choices: List[str]
    answer_idx: int

    def __len__(self):
        return len(self.choices)

    def get_answer(self) -> str:
        return self.choices[self.answer_idx]

    def get_parts_str(self) -> str:
        return "\n".join([str(part) for part in self.parts])
