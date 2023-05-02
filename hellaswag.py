from dataset import Dataset
from question import Question, Part


class Hellaswag(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__('hellaswag', *args, **kwargs)

    @staticmethod
    def to_question(row) -> Question:
        return Question(
            parts=[
                Part(
                    text=row['activity_label'],
                    tag='Activity'
                ),
                Part(
                    text=row['ctx_a'],
                    tag='Context'
                ),
                Part(
                    text=row['ctx_b'],
                    tag='Completion'
                )
            ],
            choices=row['endings'],
            answer_idx=int(row['label'])
        )

    @staticmethod
    def to_answer(answer_idx: int):
        return answer_idx
