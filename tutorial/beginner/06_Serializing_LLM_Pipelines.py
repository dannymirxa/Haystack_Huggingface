# link: https://haystack.deepset.ai/tutorials/29_serializing_pipelines

# 1. Creating a Simple Pipeline

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
# from haystack.components.generators.chat import HuggingFaceLocalChatGenerator

from chat_generator import Hugging_Face_Chat_Generator

template = [
    ChatMessage.from_user(
        """
        Please create a summary about the following topic:
        {{ topic }}
        """
            )
        ]

builder = ChatPromptBuilder(template=template)

def modified_Hugging_Face_Chat_Generator():
    chat_generator = Hugging_Face_Chat_Generator()
    chat_generator.generation_kwargs = {"max_tokens": 150}
    return chat_generator

llm = modified_Hugging_Face_Chat_Generator()

pipeline = Pipeline()
pipeline.add_component(name="builder", instance=builder)
pipeline.add_component(name="llm", instance=llm)

pipeline.connect("builder.prompt", "llm.messages")

topic = "Climate change"
result = pipeline.run(data={"builder": {"topic": topic}})
# print(result["llm"]["replies"][0].text)

# 2. Serialize the Pipeline to YAML

yaml_pipeline = pipeline.dumps()

# print(yaml_pipeline)

# 3. Editing a Pipeline in YAML

yaml_pipeline = """
components:
  builder:
    init_parameters:
      required_variables: null
      template:
      - _content:
        - text: "Please translate the following to French: \n{{ sentence }}\n"
        _meta: {}
        _name: null
        _role: user
      variables: null
    type: haystack.components.builders.chat_prompt_builder.ChatPromptBuilder
  llm:
    init_parameters:
      api_params:
        model: microsoft/Phi-3.5-mini-instruct
      api_type: serverless_inference_api
      generation_kwargs:
        max_tokens: 150
      streaming_callback: null
      token:
        env_vars:
        - HF_API_TOKEN
        strict: true
        type: env_var
      tools: null
    type: haystack.components.generators.chat.hugging_face_api.HuggingFaceAPIChatGenerator
connections:
- receiver: llm.messages
  sender: builder.prompt
max_runs_per_component: 100
metadata: {}
"""

# 4. Deseriazling a YAML Pipeline back to Python

from haystack import Pipeline

new_pipeline = Pipeline.loads(yaml_pipeline)

result = new_pipeline.run(data={"builder": {"sentence": "I love capybaras"}})

print(result['llm']['replies'][0].text)