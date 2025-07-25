from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
import os

class Text2Vector:
    def __init__(self, file_path="book-contents"):
        self.file_path = file_path
        self.persist_directory = "Chroma-db"
        self.collection_name = "Bangla-book"
        self.initialize_embeddings()
        # self.embeddings = OllamaEmbeddings(model='llama3:')

    
    def load_as_document(self):
        glob_pattern = "*.txt"
        loader = DirectoryLoader(
            path=self.file_path,
            glob=glob_pattern,
            loader_cls=TextLoader
        )

        self.documets = loader.load()

    
    def initialize_embeddings(self):
        # model_name = "sentence-transformers/all-mpnet-base-v2"
        # model_kwargs = {'device': 'cuda'}
        # encode_kwargs = {'normalize_embeddings': False}
        # self.hf_embeddings = HuggingFaceEmbeddings(
        #     model_name=model_name,
        #     model_kwargs=model_kwargs,
        #     encode_kwargs=encode_kwargs
        # )

        self.gpt4_all_embeddings = GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf")


    def text_splitting(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
            strip_whitespace=True
        )

        self.all_splits = splitter.split_documents(self.documets)
        len(self.all_splits)

    
    def text_embedding(self):
        self.embedded = [self.gpt4_all_embeddings.embed_query(split.page_content) for split in self.all_splits]

    
    def create_vectorstore(self):
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.gpt4_all_embeddings,
            persist_directory=self.persist_directory
        )

        ids = [str(i) for i in range(len(self.all_splits))]

        vector_ids = self.vector_store.add_documents(documents=self.all_splits, ids=ids)

    
    def if_db_exists(self):
        if os.path.isdir(self.persist_directory):
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.gpt4_all_embeddings
            )
        else:
            self.__call__()

        self.retriever = self.vector_store.as_retriever()

    
    def similarity_search(self, query):
        self.if_db_exists()
        results = self.vector_store.similarity_search(query)
        return results


    def __call__(self, *args, **kwds):
        self.load_as_document()
        self.text_splitting()
        self.text_embedding()
        self.create_vectorstore()


if __name__ == '__main__':
    file_path = "book-contents"
    text_loader = Text2Vector(file_path)
    text_loader()
