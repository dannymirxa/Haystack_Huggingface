# Use a pipeline as a high-level helper
import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

from huggingface_hub import InferenceClient

client = InferenceClient(
	provider="hf-inference",
	api_key=HUGGINGFACE_TOKEN
)

messages = [
	{
		"role": "user",
		"content": "What is the capital of France?"
	}
]

completion = client.chat.completions.create(
    model="microsoft/Phi-3.5-mini-instruct", 
	messages=messages, 
	max_tokens=500,
)

print(completion.choices[0].message)