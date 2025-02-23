# link: https://haystack.deepset.ai/tutorials/27_first_rag_pipeline

import os
from dotenv import load_dotenv

load_dotenv(".env")
HF_API_TOKEN = os.getenv('HF_API_TOKEN')

# 1. Initializing the DocumentStore
# https://haystack.deepset.ai/integrations?type=Document+Store
from haystack.document_stores.in_memory import InMemoryDocumentStore
document_store = InMemoryDocumentStore()

# Fetching and Indexing Documents

# 2. Fetch the Data
from datasets import load_dataset
from haystack import Document

dataset = load_dataset('bilgeyucel/seven-wonders', split="train")
# dataset = load_dataset('FreedomIntelligence/medical-o1-reasoning-SFT', "en", split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
# docs = [Document(content=doc["Response"], meta=doc["Question"]) for doc in dataset]

# 3. Initalize a Document Embedder
from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
from haystack.utils import Secret
from haystack.utils.hf import HFGenerationAPIType
from haystack.dataclasses import Document

doc = Document(content="I love pizza!")

doc_embedder = HuggingFaceAPIDocumentEmbedder(api_type="serverless_inference_api",
                                                    # Pick a text embeddings model: https://huggingface.co/models?other=text-embeddings-inference&sort=trending
                                                    api_params={"model": "sentence-transformers/all-MiniLM-L6-v2"},
                                                    token=Secret.from_env_var("HF_API_TOKEN"))

# 4. Write Documents to the DocumentStore
docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

# Building the RAG Pipeline

# 5. Initialize a Text Embedder
from haystack.components.embedders import HuggingFaceAPITextEmbedder
from haystack.utils import Secret
text_embedder = HuggingFaceAPITextEmbedder(api_type="serverless_inference_api",
                                           api_params={"model": "sentence-transformers/all-MiniLM-L6-v2"},
                                           token=Secret.from_env_var("HF_API_TOKEN"))

# 6. Initialize the Retriever
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

retriever = InMemoryEmbeddingRetriever(document_store)

# 7. Define a Template Prompt
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

template = [
    ChatMessage.from_user(
        """
        Given the following information, answer the question.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """
            )
        ]

prompt_builder = ChatPromptBuilder(template=template)

# 8. Initialize a ChatGenerator
from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack.utils.hf import HFGenerationAPIType

chat_generator = HuggingFaceAPIChatGenerator(api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                                        api_params={"model": "microsoft/Phi-3.5-mini-instruct"},
                                        token=Secret.from_env_var("HF_API_TOKEN"))

# 9. Build the Pipeline
from haystack import Pipeline

from haystack import Pipeline

basic_rag_pipeline = Pipeline()
# Add components to your pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", chat_generator)

# Now, connect the components to each other
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder")
basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

# Asking a Question

question = "What and where is Pyramid of Giza?"

response = basic_rag_pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})

print(response["llm"]["replies"][0].text)