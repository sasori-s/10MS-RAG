from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
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
        self.ollama = ChatOllama(
            model="gemma3:4b",            
        )

        self.openai = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("GPT_API"),
        )

        self.gemini = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            convert_system_message_to_human=True,
            google_api_key=os.getenv("GEMINI_API")
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
    def __init__(self, book_path, question, extra_info=None):
        self.question = question
        Text2Vector.__init__(self, book_path)
        InitiateLLM.__init__(self, extra_info)


    def retrieve(self, state: State):
        self.retrieved_docs = self.similarity_search(self.question)
        # state["context"] = self.retrieved_docs
        state.context = self.retrieved_docs
        

    # @tool(response_format="content_and_artifact")
    def generate(self, state: State):
        """Retrieve information related to a query."""
        state.question = self.question
        docs_content = "\n\n".join(doc.page_content for doc in state.context)
        message = self.few_shot_prompt.invoke({
            "context": docs_content,
            "question": state.question,
        })
        response = self.geminiß.invoke(message)
        state.answer = response
        return response
    

    def __call__(self):
        self.retrieve(State)
        self.generate(State)
    

if __name__ == '__main__':
    retriever = Retriever("book-contents", "বিয়েতে কল্লানীর মতামত কি ছিলো?")
    retriever()

