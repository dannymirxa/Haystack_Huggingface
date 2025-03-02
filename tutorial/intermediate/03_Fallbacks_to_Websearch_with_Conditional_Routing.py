# https://haystack.deepset.ai/tutorials/36_building_fallbacks_with_conditional_routing

import os
from dotenv import load_dotenv

load_dotenv(".env")
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
SERPERDEV_API_KEY = os.getenv('SERPERDEV_API_KEY')

from datasets import load_dataset
from haystack.dataclasses import Document

# Creating a Document

# Phi 3.5 Has max of 32064 tokens
# https://huggingface.co/microsoft/Phi-3.5-mini-instruct#:~:text=Phi%2D3.5%2Dmini%2DInstruct%20supports%20a%20vocabulary%20size%20of,to%20the%20model's%20vocabulary%20size.
documents = [
    Document(
        content="""Munich, the vibrant capital of Bavaria in southern Germany, exudes a perfect blend of rich cultural
                                heritage and modern urban sophistication. Nestled along the banks of the Isar River, Munich is renowned
                                for its splendid architecture, including the iconic Neues Rathaus (New Town Hall) at Marienplatz and
                                the grandeur of Nymphenburg Palace. The city is a haven for art enthusiasts, with world-class museums like the
                                Alte Pinakothek housing masterpieces by renowned artists. Munich is also famous for its lively beer gardens, where
                                locals and tourists gather to enjoy the city's famed beers and traditional Bavarian cuisine. The city's annual
                                Oktoberfest celebration, the world's largest beer festival, attracts millions of visitors from around the globe.
                                Beyond its cultural and culinary delights, Munich offers picturesque parks like the English Garden, providing a
                                serene escape within the heart of the bustling metropolis. Visitors are charmed by Munich's warm hospitality,
                                making it a must-visit destination for travelers seeking a taste of both old-world charm and contemporary allure."""
    )
]

# dataset = load_dataset('bilgeyucel/seven-wonders', split="train")
# documents = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

# Creating the Initial Pipeline Components

from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack.utils import Secret
from haystack.utils.hf import HFGenerationAPIType

prompt_template = [
    ChatMessage.from_user(
        """
        Answer the following query given the documents.
        If the answer is not contained within the documents reply with 'no_answer'
        Query: {{query}}
        Documents:
        {% for document in documents %}
        {{document.content}}
        {% endfor %}
        """
            )
        ]

prompt_builder = ChatPromptBuilder(template=prompt_template)
llm = HuggingFaceAPIChatGenerator(api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                                        api_params={"model": "microsoft/Phi-3.5-mini-instruct"},
                                        token=Secret.from_env_var("HF_API_TOKEN"))

# Initializing the Web Search Components

from haystack.components.websearch.serper_dev import SerperDevWebSearch

prompt_for_websearch = [
    ChatMessage.from_user(
        """
        Answer the following query given the documents retrieved from the web.
        Your answer shoud indicate that your answer was generated from websearch.

        Query: {{query}}
        Documents:
        {% for document in documents %}
        {{document.content}}
        {% endfor %}
        """
            )
        ]

websearch = SerperDevWebSearch(api_key=Secret.from_env_var("SERPERDEV_API_KEY"))
prompt_builder_for_websearch = ChatPromptBuilder(template=prompt_for_websearch)
llm_for_websearch = HuggingFaceAPIChatGenerator(api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                                        api_params={"model": "microsoft/Phi-3.5-mini-instruct"},
                                        token=Secret.from_env_var("HF_API_TOKEN"))

# Creating the ConditionalRouter

from haystack.components.routers import ConditionalRouter

routes = [
    {
        "condition": "{{'no_answer' in replies[0].text}}",
        "output": "{{query}}",
        "output_name": "go_to_websearch",
        "output_type": str,
    },
    {
        "condition": "{{'no_answer' not in replies[0].text}}",
        "output": "{{replies[0].text}}",
        "output_name": "answer",
        "output_type": str,
    },
]

router = ConditionalRouter(routes)

# Building the Pipeline

from haystack import Pipeline

pipe = Pipeline()
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)
pipe.add_component("router", router)
pipe.add_component("websearch", websearch)
pipe.add_component("prompt_builder_for_websearch", prompt_builder_for_websearch)
pipe.add_component("llm_for_websearch", llm_for_websearch)

pipe.connect("prompt_builder.prompt", "llm.messages")
pipe.connect("llm.replies", "router.replies")
pipe.connect("router.go_to_websearch", "websearch.query")
pipe.connect("router.go_to_websearch", "prompt_builder_for_websearch.query")
pipe.connect("websearch.documents", "prompt_builder_for_websearch.documents")
pipe.connect("prompt_builder_for_websearch", "llm_for_websearch")

pipe.draw("pipe.png")

query = "Where is Munich?"

result = pipe.run({"prompt_builder": {"query": query, "documents": documents}, "router": {"query": query}})

# Print the `answer` coming from the ConditionalRouter
print(result["router"]["answer"])

query = "How many people live in Munich?"

result = pipe.run({"prompt_builder": {"query": query, "documents": documents}, "router": {"query": query}})

# Print the `replies` generated using the web searched Documents
print(result["llm_for_websearch"]["replies"])

