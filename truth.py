from typing import List

import numpy as np

from dataset.cosmos_qa import CosmosQA
from dataset.hellaswag import Hellaswag
from dataset.race import Race
from question import Question


def to_answers(questions: List[Question]) -> List[int]:
    answers = []
    for question in questions:
        answers.append(question.answer_idx)
    return answers


if __name__ == '__main__':
    cosmos_qa = CosmosQA(split='validation')
    race = Race('middle', split='validation')
    hellaswag = Hellaswag(split='validation')

    np.savetxt('cosmos_qa_truth.txt', X=np.array(to_answers(cosmos_qa.to_questions())))
    np.savetxt('race_truth.txt', X=np.array(to_answers(race.to_questions())))
    np.savetxt('hellaswag_truth.txt', X=np.array(to_answers(hellaswag.to_questions())))
