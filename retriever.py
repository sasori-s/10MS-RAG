from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from typing_extensions import List, TypedDict
import os
from vectorizer import Text2Vector

class State(TypedDict):
    question: str
    context: List[Document]
    asnwer: str


class InitiateLLM:
    def __init__(self, extra_info=None):
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

    
    def generate_templates(self):
        examples = [
            {
                "user_question" : " অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
                "expected_answer" : "শুম্ভুনাথ"
                },

            {
                "user_question" : " কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
                "expected_answer" : "মামাকে"
                },

            {
                "user_question" : "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
                "expected_answer" : "১৫ বছর"
                }
        ]

        example_template = """
            User: {user_question}
            AI: {expected_answer}
        """

        example_prompt = PromptTemplate(
            input_variables=["user_question", "expected_answer"],
            template=example_template
        )

        self.few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="The following are the instructions of question-answering interactions: Answer them with the given content",
            suffix="Now, answer the following questions based on the retrieved content which is {context} \nUser:{question}\nAI:",
            input_variables=["context","question"]
        )


class Retriever(Text2Vector, InitiateLLM):
    def __init__(self, book_path, extra_info=None):
        Text2Vector.__init__(self, book_path)
        InitiateLLM.__init__(self, extra_info)

    def retrieve(self, state: State):
        self.retrieved_docs = self.similarity_search()
        # state["context"] = self.retrieved_docs
        state.context = self.retrieved_docs

    def generate(self, state: State):
        state.question = "অনুপমের মামার নাম কি ছিলো ?"
        docs_content = "\n\n".join(doc.page_content for doc in state.context)
        message = self.few_shot_prompt.invoke({
            "context": docs_content,
            "question": state.question,
        })
        response = self.llm.invoke(message)
        state.answer = response
        return response

    def __call__(self):
        self.retrieve(State)
        self.generate(State)
    

if __name__ == '__main__':
    retriever = Retriever("book-contents")
    retriever()