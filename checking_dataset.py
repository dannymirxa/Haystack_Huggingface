from datasets import load_dataset
from haystack import Document


# dataset = load_dataset('bilgeyucel/seven-wonders', split="train")
# dataset = load_dataset('rajpurkar/squad', split="train", streaming=True)

# dataset = load_dataset('FreedomIntelligence/medical-o1-reasoning-SFT', "en", split="train")
# docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
# docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
# docs = [(doc['context'], doc['question']) for doc in dataset][0:2]
# docs = [Document(content=doc["Response"], meta=doc["Question"]) for doc in dataset][0:2]
# docs = [doc["meta"] for doc in dataset]

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# tokenized_dataset = dataset.map(lambda x: tokenizer(x['context']), batched=True)

# print(docs)

# print(tokenized_dataset)

import json
from typing import List
from pydantic import BaseModel


class City(BaseModel):
    name: str
    country: str
    population: int


class CitiesData(BaseModel):
    cities: List[City]

# json_schema = CitiesData.model_json_schema() 
# json_schema = json.dumps(json_schema, indent=2)
json_schema = CitiesData.schema_json(indent=2)
print(json_schema)