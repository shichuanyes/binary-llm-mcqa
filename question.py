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

    def get_natural_prompt(self) -> str:
        prompt = f"self.get_parts_str()\n"
        for i, choice in enumerate(self.choices):
            prompt += f"{idx_to_letter(i)}. {choice}\n"
        prompt += "Answer:"
        return prompt

    def get_binary_prompts(self) -> List[str]:
        prompts = []
        for choice in self.choices:
            prompt = f"self.get_parts_str()\n" \
                     f"Answer: {choice}\n" \
                     f"Correct:"
            prompts.append(prompt)
        return prompts


