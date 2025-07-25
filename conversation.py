from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from retriever import Retriever, InitiateLLM
from langchain.chains import ConversationChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

class LLMConversation(Retriever):
    def __init__(self):
        Retriever().__init__("book-contets", "গহনা গুলা কিসের তৈরি ছিলো?")
        self.output_parser = StrOutputParser()
        self.chat_history = []

    def defining_memory_prompts(self):
        instruction_to_system = """
        Given a chat history and the lastest user question
        which might reference context in the chat history, formulate a standalone question
        which can be understood without the chat history. Do NOT answer the question,
        just reformulate it if neeeded and other return it as is.
        """

        question_maker_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", instruction_to_system),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ]
        )

        self.question_chain = question_maker_prompt | self.ollama | StrOutputParser()

        qa_system_prompt = """You are an assistant for question-answering tasks.\
        Use the following peices of retrieved context to answer the question. \
        IF you don't know the asnwer, provide a summary of the content. Do not generate your answer.\

        {context}
        """
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ]
        )

    
    def contextualized_question(self, input: dict):
        if input.get("chat_history"):
            return self.question_chain
        else:
            return input['question']
        
    
    def chat_retrieving(self):
        self.retriever_chain = RunnablePassthrough.assign(
            context=self.contextualized_question | self.retriever
        )

        self.rag_chain = (
            self.retriever_chain
            | self.qa_prompt
            | self.ollama
            | self.output_parser
        )

        question = "Ask something about onupom"

        ai_message = self.rag_chain.invoke({"question": question, "chat_history": self.chat_history})
        self.chat_history.extend([HumanMessage(content=question), ai_message])


if __name__ == '__main__':
    conversation = LLMConversation()
    conversation.defining_memory_prompts()