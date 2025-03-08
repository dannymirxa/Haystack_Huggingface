# link: https://haystack.deepset.ai/tutorials/33_hybrid_retrieval

# 1. Initializing the DocumentStore

import os
from dotenv import load_dotenv
from haystack.utils import Secret

load_dotenv(".env")
HF_API_TOKEN = os.getenv('HF_API_TOKEN')

from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()

# 2. Fetching and Processing Documents

from datasets import load_dataset
from haystack import Document

dataset = load_dataset("anakin87/medrag-pubmed-chunk", split="train")

docs = []
for doc in dataset:
    docs.append(
        Document(content=doc["contents"], meta={"title": doc["title"], "abstract": doc["content"], "pmid": doc["id"]})
    )

# 3. Indexing Documents with a Pipeline

from haystack.components.writers import DocumentWriter
# from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack import Pipeline
from haystack.utils import ComponentDevice

document_splitter = DocumentSplitter(split_by="word", split_length=512, split_overlap=32)
# document_embedder = SentenceTransformersDocumentEmbedder(
#     model="BAAI/bge-small-en-v1.5", device=ComponentDevice.from_str("cuda:0")
# )
from custom_HF.doc_embedder import Hugging_Face_Document_Embedder
document_embedder = Hugging_Face_Document_Embedder('BAAI/bge-small-en-v1.5')
document_writer = DocumentWriter(document_store)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("document_splitter", document_splitter)
indexing_pipeline.add_component("document_embedder", document_embedder)
indexing_pipeline.add_component("document_writer", document_writer)

indexing_pipeline.connect("document_splitter", "document_embedder")
indexing_pipeline.connect("document_embedder", "document_writer")

indexing_pipeline.run({"document_splitter": {"documents": docs}})

# 4. Creating a Pipeline for Hybrid Retrieval

# 4.1 Initialize Retrievers and the Embedder

from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
# from haystack.components.embedders import SentenceTransformersTextEmbedder
from custom_HF.text_embedder import Hugging_Face_Text_Embedder

# text_embedder = SentenceTransformersTextEmbedder(
#     model="BAAI/bge-small-en-v1.5", device=ComponentDevice.from_str("cuda:0")
# )

text_embedder = Hugging_Face_Text_Embedder('BAAI/bge-small-en-v1.5')

embedding_retriever = InMemoryEmbeddingRetriever(document_store)
bm25_retriever = InMemoryBM25Retriever(document_store)

# 4.2 Join Retrieval Results

from haystack.components.joiners import DocumentJoiner

document_joiner = DocumentJoiner()

# 4.3 Rank the Results

from haystack.components.rankers import TransformersSimilarityRanker

ranker = TransformersSimilarityRanker(token=Secret.from_env_var("HF_API_TOKEN"), model="BAAI/bge-reranker-base")

# 4.4 Create the Hybrid Retrieval Pipeline

from haystack import Pipeline

hybrid_retrieval = Pipeline()
hybrid_retrieval.add_component("text_embedder", text_embedder)
hybrid_retrieval.add_component("embedding_retriever", embedding_retriever)
hybrid_retrieval.add_component("bm25_retriever", bm25_retriever)
hybrid_retrieval.add_component("document_joiner", document_joiner)
hybrid_retrieval.add_component("ranker", ranker)

hybrid_retrieval.connect("text_embedder", "embedding_retriever")
hybrid_retrieval.connect("bm25_retriever", "document_joiner")
hybrid_retrieval.connect("embedding_retriever", "document_joiner")
hybrid_retrieval.connect("document_joiner", "ranker")

# 4.5 Visualize the Pipeline (Optional)

hybrid_retrieval.draw("hybrid-retrieval.png")

# 5. Testing the Hybrid Retrieval

query = "apnea in infants"

result = hybrid_retrieval.run(
    {"text_embedder": {"text": query}, "bm25_retriever": {"query": query}, "ranker": {"query": query}}
)

def pretty_print_results(prediction):
    for doc in prediction["documents"]:
        print(doc.meta["title"], "\t", doc.score)
        print(doc.meta["abstract"])
        print("\n", "\n")

pretty_print_results(result["ranker"])
