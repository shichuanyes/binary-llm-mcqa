import numpy as np

from dataset.cosmos_qa import CosmosQA
from dataset.race import Race

if __name__ == '__main__':
    # val = CosmosQA(split='validation')
    val = Race('middle', split='validation')
    # val = Hellaswag(split='validation')
    answers = np.loadtxt('race_result_curie_binary.txt')

    questions = val.to_questions()
    count = 0
    for i, question in enumerate(questions):
        if answers[i] == question.answer_idx:
            count += 1
    accuracy = count / len(questions)
    print(f'accuracy={accuracy}')
