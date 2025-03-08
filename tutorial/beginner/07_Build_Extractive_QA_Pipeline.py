# link: https://haystack.deepset.ai/tutorials/34_extractive_qa_pipeline

# 1. Load data into the DocumentStore

from datasets import load_dataset
from haystack import Document
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from custom_HF.doc_embedder import Hugging_Face_Document_Embedder
from haystack.components.writers import DocumentWriter


dataset = load_dataset("bilgeyucel/seven-wonders", split="train")

documents = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

# model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"



document_store = InMemoryDocumentStore()

indexing_pipeline = Pipeline()


indexing_pipeline.add_component(instance=Hugging_Face_Document_Embedder('sentence-transformers/multi-qa-mpnet-base-dot-v1'), name="embedder")
indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
indexing_pipeline.connect("embedder.documents", "writer.documents")

indexing_pipeline.run({"documents": documents})

# 2. Build an Extractive QA Pipeline

retriever = InMemoryEmbeddingRetriever(document_store=document_store)
reader = ExtractiveReader()
reader.warm_up()

extractive_qa_pipeline = Pipeline()

from custom_HF.text_embedder import Hugging_Face_Text_Embedder
extractive_qa_pipeline.add_component(instance= Hugging_Face_Text_Embedder('sentence-transformers/multi-qa-mpnet-base-dot-v1'), name="embedder")
extractive_qa_pipeline.add_component(instance=retriever, name="retriever")
extractive_qa_pipeline.add_component(instance=reader, name="reader")

extractive_qa_pipeline.connect("embedder.embedding", "retriever.query_embedding")
extractive_qa_pipeline.connect("retriever.documents", "reader.documents")

query = "Who was Pliny the Elder?"
result = extractive_qa_pipeline.run(
    data={"embedder": {"text": query}, "retriever": {"top_k": 3}, "reader": {"query": query, "top_k": 2}}
)

print(result['reader']['answers'])

# 3. ExtractiveReader: a closer look

