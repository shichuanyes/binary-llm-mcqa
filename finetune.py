import json
from typing import List, Dict, Type

from tqdm import tqdm

from cosmos_qa import CosmosQA
from dataset import Dataset
from method import Method
from natural_method import NaturalMethod
from race import Race


def get_ft_jsonl(dataset: Dataset, method: Type[Method]) -> List[Dict[str, str]]:
    questions = dataset.to_questions()
    print("Converted to list of Questions")
    result = []
    for question in tqdm(questions):
        result += method.prepare_finetune(question)
    return result


if __name__ == '__main__':
    # cosmos_qa = CosmosQA(split='train')
    race = Race('middle', split='train')
    print("Generated finetune data")
    # ft = get_ft_jsonl(cosmos_qa, NaturalMethod)
    ft = get_ft_jsonl(race, NaturalMethod)
    with open('race.jsonl', 'w') as file:
        for entry in ft:
            json.dump(entry, file)
            file.write('\n')
