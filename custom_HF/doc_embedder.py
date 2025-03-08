import os
from dotenv import load_dotenv

from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
from haystack.utils import Secret

load_dotenv(".env")

def Hugging_Face_Document_Embedder(model: str) -> HuggingFaceAPIDocumentEmbedder:
    HF_API_TOKEN = os.getenv('HF_API_TOKEN')
    doc_embedder = HuggingFaceAPIDocumentEmbedder(api_type="serverless_inference_api",
                                                # Pick a text embeddings model: https://huggingface.co/models?other=text-embeddings-inference&sort=trending
                                                api_params={"model": model},
                                                token=Secret.from_env_var("HF_API_TOKEN")
                                                )
    return doc_embedder