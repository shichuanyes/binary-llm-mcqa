from dataset import Dataset
from question import Question, Part
from utils import letter_to_idx, idx_to_letter


class Race(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__('race', *args, **kwargs)

    @staticmethod
    def to_question(row) -> Question:
        return Question(
            parts=[
                Part(
                    text=row['article'],
                    tag='Article'
                ),
                Part(
                    text=row['question'],
                    tag='Question'
                )
            ],
            choices=row['options'],
            answer_idx=letter_to_idx(row['answer'])
        )

    @staticmethod
    def to_answer(answer_idx: int):
        return idx_to_letter(answer_idx)