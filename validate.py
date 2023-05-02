from typing import Callable, List, Tuple

import numpy as np
import openai
from tqdm import tqdm

from api_key import api_key
from cosmos_qa import CosmosQA
from dataset import Dataset
from models import curie_finetune
from natural_method import NaturalMethod


def evaluate(dataset: Dataset, predict: Callable) -> Tuple[float, List[int]]:
    questions = dataset.to_questions()
    print("Converted to list of Questions")
    result = []
    count = 0
    for question in tqdm(questions):
        pred = predict(question)
        result.append(pred)
        if pred == question.answer_idx:
            count += 1

    return count / len(questions), result


if __name__ == '__main__':
    cosmos_qa_val = CosmosQA(split='validation')
    openai.api_key = api_key
    accuracy, answers = evaluate(cosmos_qa_val, lambda question: NaturalMethod.ask(question, model=curie_finetune))

    np.savetxt('cosmos_qa_result_naive.txt', X=np.array(answers))
    print(f'accuracy={accuracy}')
