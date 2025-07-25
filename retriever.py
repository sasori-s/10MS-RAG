from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
import os
from vectorizer import Text2Vector

class State(TypedDict):
    question: str
    context: List[Document]
    asnwer: str


class InitiateLLM:
    def __init__(self, extra_info=[]):
        self.extra_info = extra_info
        self.llm = ChatOllama(
            model="gemma3:4b",            
        )

        self.llm2 = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("GPT_API"),
        )

        self.generate_templates()


class Retriever(Text2Vector):
    def __init__(self, book_path):
        super(Retriever, self).__init__(book_path)

    
    def retrieve(self, state: State):
        self.retrieved_docs = self.similarity_search()
        state["context"] = self.retrieved_docs
    

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    


if __name__ == '__main__':
    retriever = Retriever("book-contents")
    retriever.retrieve(State)