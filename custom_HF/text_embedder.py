import os
from dotenv import load_dotenv

from haystack.components.embedders import HuggingFaceAPITextEmbedder
from haystack.utils import Secret

load_dotenv(".env")

def Hugging_Face_Text_Embedder(model: str) -> HuggingFaceAPITextEmbedder:
    HF_API_TOKEN = os.getenv('HF_API_TOKEN')
    text_embedder = HuggingFaceAPITextEmbedder(api_type="serverless_inference_api",
                                                # Pick a text embeddings model: https://huggingface.co/models?other=text-embeddings-inference&sort=trending
                                                api_params={"model": model},
                                                token=Secret.from_env_var("HF_API_TOKEN")
                                                )
    return text_embedder