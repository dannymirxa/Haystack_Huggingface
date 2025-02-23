import os
from dotenv import load_dotenv

load_dotenv(".env")

HF_API_TOKEN = os.getenv('HF_API_TOKEN')

from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack import Pipeline
from haystack.utils import Secret
from haystack.utils.hf import HFGenerationAPIType

# no parameter init, we don't use any runtime template variables
prompt_builder = ChatPromptBuilder()
llm = HuggingFaceAPIChatGenerator(api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                                  api_params={"model": "microsoft/Phi-3.5-mini-instruct"},
                                  token=Secret.from_env_var("HF_API_TOKEN"))

pipe = Pipeline()
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)
pipe.connect("prompt_builder.prompt", "llm.messages")

topic = "Capital"
messages = [ChatMessage.from_system("Respond in short but methodical, straight to the point English."),
            ChatMessage.from_user("Tell me about Kuala Lumpur")]

result = pipe.run(data={"prompt_builder": {"template_variables":{"topic": topic}, "template": messages}})

print(result)