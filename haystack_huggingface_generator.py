import os
from dotenv import load_dotenv

load_dotenv(".env")

HF_API_TOKEN = os.getenv('HF_API_TOKEN')

from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
# HuggingFaceAPIGenerator can be used to generate text using different Hugging Face APIs
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document

docstore = InMemoryDocumentStore()
docstore.write_documents([Document(content="Rome is the capital of Italy"), Document(content="Paris is the capital of France")])

query = "What is the capital of France?"

template = """
Given the following information, answer the question.

Context: 
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ query }}?
"""

generator = HuggingFaceAPIGenerator(api_type="serverless_inference_api",
                                    api_params={"model": "microsoft/Phi-3.5-mini-instruct"},
                                    token=Secret.from_token(HF_API_TOKEN))

pipe = Pipeline()

pipe.add_component("retriever", InMemoryBM25Retriever(document_store=docstore))
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", generator)
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

res=pipe.run({
    "prompt_builder": {
        "query": query
    },
    "retriever": {
        "query": query
    }
})

print(res)

