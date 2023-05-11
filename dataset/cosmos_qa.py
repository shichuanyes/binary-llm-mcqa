from dataset.dataset import Dataset
from question import Question, Part


class CosmosQA(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__('cosmos_qa', *args, **kwargs)

    @staticmethod
    def to_question(row) -> Question:
        return Question(
            parts=[
                Part(
                    text=row['context'],
                    tag='Context'
                ),
                Part(
                    text=row['question'],
                    tag='Question'
                )
            ],
            choices=[
                row['answer0'],
                row['answer1'],
                row['answer2'],
                row['answer3']
            ],
            answer_idx=row['label']
        )

    @staticmethod
    def to_answer(answer_idx: int):
        return answer_idx
