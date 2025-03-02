# link: https://haystack.deepset.ai/tutorials/42_sentence_window_retriever

from haystack import Document
from haystack.components.preprocessors import DocumentSplitter

# 1. Getting started with Sentence-Window Retrieval

splitter = DocumentSplitter(split_length=1, split_overlap=0, split_by="period")

text = (
    "Paul fell asleep to dream of an Arrakeen cavern, silent people all around  him moving in the dim light "
    "of glowglobes. It was solemn there and like a cathedral as he listened to a faint sound—the "
    "drip-drip-drip of water. Even while he remained in the dream, Paul knew he would remember it upon "
    "awakening. He always remembered the dreams that were predictions. The dream faded. Paul awoke to feel "
    "himself in the warmth of his bed—thinking thinking. This world of Castle Caladan, without play or "
    "companions his own age,  perhaps did not deserve sadness in farewell. Dr Yueh, his teacher, had "
    "hinted  that the faufreluches class system was not rigidly guarded on Arrakis. The planet sheltered "
    "people who lived at the desert edge without caid or bashar to command them: will-o’-the-sand people "
    "called Fremen, marked down on no  census of the Imperial Regate."
)

doc = Document(content=text)
docs = splitter.run([doc])

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

doc_store = InMemoryDocumentStore()
doc_store.write_documents(docs["documents"], policy=DuplicatePolicy.OVERWRITE)

from haystack.components.retrievers import SentenceWindowRetriever

retriever = SentenceWindowRetriever(document_store=doc_store, window_size=2)
result = retriever.run(retrieved_documents=[docs["documents"][4]])

# 2. Create a Keyword Retrieval Pipeline with Sentence-Window Retrieval

# 2.1 Reading the dataset

from typing import List
import csv
from haystack import Document

def read_documents(file: str) -> List[Document]:
    with open(file, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader, None)  # skip the headers
        documents = []
        for row in reader:
            category = row[0].strip()
            title = row[2].strip()
            text = row[3].strip()
            documents.append(Document(content=text, meta={"category": category, "title": title}))

    return documents

# from pathlib import Path
# import requests

# doc = requests.get("https://raw.githubusercontent.com/amankharwal/Website-data/master/bbc-news-data.csv")

# datafolder = Path("data")
# datafolder.mkdir(exist_ok=True)
# with open(datafolder / "bbc-news-data.csv", "wb") as f:
#     for chunk in doc.iter_content(512):
#         f.write(chunk)

docs = read_documents("data/bbc-news-data.csv")

# 2.2 Indexing the documents

from haystack import Document, Pipeline
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

doc_store = InMemoryDocumentStore()

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("splitter", DocumentSplitter(split_length=1, split_overlap=0, split_by="sentence"))
indexing_pipeline.add_component("writer", DocumentWriter(document_store=doc_store, policy=DuplicatePolicy.OVERWRITE))

indexing_pipeline.connect("splitter", "writer")

indexing_pipeline.run({"documents": docs})

# 2.3 Build a Sentence-Window Retrieval Pipeline

from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.retrievers import SentenceWindowRetriever

sentence_window_pipeline = Pipeline()

sentence_window_pipeline.add_component("bm25_retriever", InMemoryBM25Retriever(document_store=doc_store))
sentence_window_pipeline.add_component("sentence_window__retriever", SentenceWindowRetriever(doc_store, window_size=2))

sentence_window_pipeline.connect("bm25_retriever.documents", "sentence_window__retriever.retrieved_documents")

# 2.4 Putting it all together

result = sentence_window_pipeline.run(
    data={"bm25_retriever": {"query": "phishing attacks", "top_k": 1}}, include_outputs_from={"bm25_retriever"}
)

print(result["bm25_retriever"]["documents"])

print(result["sentence_window__retriever"]["context_windows"])

print(result["sentence_window__retriever"]["context_documents"])