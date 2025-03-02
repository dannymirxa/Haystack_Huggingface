import os
from dotenv import load_dotenv

from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack.utils import Secret
from haystack.utils.hf import HFGenerationAPIType

load_dotenv(".env")

def Hugging_Face_Chat_Generator():
    HF_API_TOKEN = os.getenv('HF_API_TOKEN')
    chat_generator = HuggingFaceAPIChatGenerator(api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                                            api_params={"model": "microsoft/Phi-3.5-mini-instruct"},
                                            token=Secret.from_env_var("HF_API_TOKEN")
                                            )
    return chat_generator