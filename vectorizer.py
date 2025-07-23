from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import DirectoryLoader

class Text2Document:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_as_document(self):
        glob_pattern = "*.txt"
        loader = DirectoryLoader(
            path=self.file_path,
            glob=glob_pattern,
            loader_cls=TextLoader
        )

        documets = loader.load()

        for doc in documets:
            print(f"Page content {doc.page_content}")


if __name__ == '__main__':
    file_path = "book-contents"
    text_loader = Text2Document(file_path)
    text_loader.load_as_document()
