import json
from typing import List, Dict, Type

from tqdm import tqdm

from dataset.cosmos_qa import CosmosQA
from dataset.dataset import Dataset
from dataset.hellaswag import Hellaswag
from dataset.race import Race
from method.binary_method import BinaryMethod
from method.method import Method
from method.natural_method import NaturalMethod


def get_ft_jsonl(dataset: Dataset, method: Type[Method]) -> List[Dict[str, str]]:
    questions = dataset.to_questions()
    print("Converted to list of Questions")
    result = []
    for question in tqdm(questions):
        result += method.prepare_finetune(question)
    return result


if __name__ == '__main__':
    # data = CosmosQA(split='train')
    # data = Race('middle', split='train')
    data = Hellaswag(split='train')
    print("Generated finetune data")
    ft = get_ft_jsonl(data, BinaryMethod)
    with open('hellaswag_binary.jsonl', 'w') as file:
        for entry in ft:
            json.dump(entry, file)
            file.write('\n')
