import os
from dotenv import load_dotenv
import json

load_dotenv(".env")

HF_API_TOKEN = os.getenv('HF_API_TOKEN')

from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack.utils.hf import HFGenerationAPIType
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

def generate_simple_response(prompt: str):
    messages = [ChatMessage.from_system("\\nYou are a helpful, respectful and honest assistant"),
                ChatMessage.from_user(prompt)]

    generator = HuggingFaceAPIChatGenerator(api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                                    api_params={"model": "microsoft/Phi-3.5-mini-instruct"},
                                    token=Secret.from_token(HF_API_TOKEN))

    result = generator.run(messages)

    # return str(result["replies"][0]).split("text='")[1].split("'", 1)[0]
    return result

# print(generate_simple_response("What is Natual Language Processing?"))
