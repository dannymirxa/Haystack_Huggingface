import os
from dotenv import load_dotenv

load_dotenv(".env")

HF_API_TOKEN = os.getenv('HF_API_TOKEN')

from haystack.components.builders import ChatPromptBuilder
# This component’s main input is a list of ChatMessage objects. ChatMessage is a data class that contains a message, a role (who generated the message, such as user, assistant, system, function), and optional metadata. For more information, check out our ChatMessage docs.
from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack import Pipeline
from haystack.utils import Secret
from haystack.utils.hf import HFGenerationAPIType

# no parameter init, we don't use any runtime template variables
prompt_builder = ChatPromptBuilder()
llm = HuggingFaceAPIChatGenerator(api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                                  api_params={"model": "microsoft/Phi-3.5-mini-instruct"},
                                  token=Secret.from_token(HF_API_TOKEN))
                                        
pipe = Pipeline()
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)
pipe.connect("prompt_builder.prompt", "llm.messages")
location = "Berlin"
messages = [ChatMessage.from_system("Always respond in German even if some input data is in other languages."),
ChatMessage.from_user("Tell me about {{location}}")]
result = pipe.run(data={"prompt_builder": {"template_variables":{"location": location}, "template": messages}})

print(result)

