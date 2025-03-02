import os
from dotenv import load_dotenv

load_dotenv(".env")
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
SERPERDEV_API_KEY = os.getenv('SERPERDEV_API_KEY')

# Learning about the HuggingFaceAPIChatGenerator

from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack.utils import Secret
from haystack.utils.hf import HFGenerationAPIType

# 1. Basic Streaming

from haystack.components.generators.utils import print_streaming_chunk

messages = [
    ChatMessage.from_system("Always respond in simple English even if some input data is in other languages."),
    ChatMessage.from_user("What's Natural Language Processing? Be brief."),
]

chat_generator = HuggingFaceAPIChatGenerator(api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                                        api_params={"model": "microsoft/Phi-3.5-mini-instruct"},
                                        token=Secret.from_env_var("HF_API_TOKEN"),
                                        streaming_callback=print_streaming_chunk
                                        )
response = chat_generator.run(messages=messages)

# Creating a Function Calling Tool from a Haystack Pipeline

# 2. Index Documents with a Pipeline

from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

documents = [
    Document(content="My name is Jean and I live in Paris."),
    Document(content="My name is Mark and I live in Berlin."),
    Document(content="My name is Giorgio and I live in Rome."),
    Document(content="My name is Marta and I live in Madrid."),
    Document(content="My name is Harry and I live in London."),
]

document_store = InMemoryDocumentStore()

indexing_pipeline = Pipeline()
indexing_pipeline.add_component(
    instance=SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"), name="doc_embedder"
)
indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="doc_writer")

indexing_pipeline.connect("doc_embedder.documents", "doc_writer.documents")

indexing_pipeline.run({"doc_embedder": {"documents": documents}})

# 3. Build a RAG Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

template = [
    ChatMessage.from_system(
        """
        Answer the questions based on the given context.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}
        Question: {{ question }}
        Answer:
        """
            )
        ]
rag_pipe = Pipeline()
rag_pipe.add_component("embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
rag_pipe.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
rag_pipe.add_component("prompt_builder", ChatPromptBuilder(template=template))
rag_pipe.add_component("llm", HuggingFaceAPIChatGenerator(api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                                        api_params={"model": "microsoft/Phi-3.5-mini-instruct"},
                                        token=Secret.from_env_var("HF_API_TOKEN")))

rag_pipe.connect("embedder.embedding", "retriever.query_embedding")
rag_pipe.connect("retriever", "prompt_builder.documents")
rag_pipe.connect("prompt_builder.prompt", "llm.messages")

# 4. Run the Pipeline
query = "Where does Mark live?"
rag_pipe.run({"embedder": {"text": query}, "prompt_builder": {"question": query}})

# 5. Convert the Haystack Pipeline into a Tool
from haystack.tools import Tool


def rag_pipeline_func(query: str):
    result = rag_pipe.run({"embedder": {"text": query}, "prompt_builder": {"question": query}})
    return {"reply": result["llm"]["replies"][0].text}


parameters = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The query to use in the search. Infer this from the user's message. It should be a question or a statement",
        }
    },
    "required": ["query"],
}

rag_pipeline_tool = Tool(
    name="rag_pipeline_tool",
    description="Get information about where people live",
    parameters=parameters,
    function=rag_pipeline_func,
)

# Creating a Tool from Function

from typing import Annotated, Literal
from haystack.tools import create_tool_from_function

WEATHER_INFO = {
    "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
    "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
    "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    "Madrid": {"weather": "sunny", "temperature": 10, "unit": "celsius"},
    "London": {"weather": "cloudy", "temperature": 9, "unit": "celsius"},
}


def get_weather(
    city: Annotated[str, "the city for which to get the weather"] = "Berlin",
    unit: Annotated[Literal["Celsius", "Fahrenheit"], "the unit for the temperature"] = "Celsius",
):
    """A simple function to get the current weather for a location."""
    if city in WEATHER_INFO:
        return WEATHER_INFO[city]
    else:
        return {"weather": "sunny", "temperature": 21.8, "unit": "fahrenheit"}


weather_tool = create_tool_from_function(get_weather)

# Running HuggingFaceAPIChatGenerator with Tools

from haystack.dataclasses import ChatMessage
from haystack.components.generators.utils import print_streaming_chunk

user_messages = [
    ChatMessage.from_system(
        "Use the tool that you're provided with. Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
    ),
    ChatMessage.from_user("Can you tell me where Mark lives?"),
]

chat_generator = HuggingFaceAPIChatGenerator(api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                                        api_params={"model": "microsoft/Phi-3.5-mini-instruct"},
                                        token=Secret.from_env_var("HF_API_TOKEN"),
                                        # streaming_callback=print_streaming_chunk
                                        )
response = chat_generator.run(messages=user_messages, tools=[rag_pipeline_tool, weather_tool])

from haystack.components.tools import ToolInvoker

tool_invoker = ToolInvoker(tools=[rag_pipeline_tool, weather_tool])

if response["replies"][0].tool_calls:
    tool_result_messages = tool_invoker.run(messages=response["replies"])["tool_messages"]
    print(f"tool result messages: {tool_result_messages}")

# Pass all the messages to the ChatGenerator with the correct order
messages = user_messages + response["replies"] + tool_result_messages
final_replies = chat_generator.run(messages=messages, tools=[rag_pipeline_tool, weather_tool])["replies"]

import gradio as gr
import json

from haystack.dataclasses import ChatMessage
from haystack.components.tools import ToolInvoker

tool_invoker = ToolInvoker(tools=[rag_pipeline_tool, weather_tool])
chat_generator = HuggingFaceAPIChatGenerator(api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API,
                                        api_params={"model": "microsoft/Phi-3.5-mini-instruct"},
                                        token=Secret.from_env_var("HF_API_TOKEN"),
                                        tools=[rag_pipeline_tool, weather_tool])
response = None
messages = [
    ChatMessage.from_system(
        "Use the tool that you're provided with. Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
    )
]


def chatbot_with_tc(message, history):
    global messages
    messages.append(ChatMessage.from_user(message))
    response = chat_generator.run(messages=messages)

    while True:
        # if OpenAI response is a function call
        if response and response["replies"][0].tool_calls:
            tool_result_messages = tool_invoker.run(messages=response["replies"])["tool_messages"]
            messages = messages + response["replies"] + tool_result_messages
            response = chat_generator.run(messages=messages)

        # Regular Conversation
        else:
            messages.append(response["replies"][0])
            break
    return response["replies"][0].text


demo = gr.ChatInterface(
    fn=chatbot_with_tc,
    type="messages",
    examples=[
        "Can you tell me where Giorgio lives?",
        "What's the weather like in Madrid?",
        "Who lives in London?",
        "What's the weather like where Mark lives?",
    ],
    title="Ask me about weather or where people live!",
    theme=gr.themes.Ocean(),
)

demo.launch()