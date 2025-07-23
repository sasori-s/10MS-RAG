from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import os

class InitiateLLM:
    def __init__(self, extra_info=[]):
        self.extra_info = extra_info
        self.llm = ChatOllama(
            model="gemma3:4b",            
        )

        self.llm2 = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("GPT4_API"),
        )

        self.generate_templates()
        