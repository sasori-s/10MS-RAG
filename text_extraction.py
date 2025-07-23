from langchain_community.document_loaders import PyPDFLoader
import asyncio
from io import StringIO
from pdfminer.layout import LAParams
from pdfminer.utils import open_filename
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from langchain_community.document_loaders import PDFPlumberLoader, PDFMinerLoader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os


class PDFLoaderRegular:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    async def loader(self):
        loader = PyPDFLoader(self.pdf_path)
        pages = []

        async for page in loader.alazy_load():
            pages.append(page)

        print(pages[0].page_content)


    def iter_text_per_page(self, pdf_file, page_numbers=None, maxpages=0, caching=True, codec='utf-8', laparams=None):
        if laparams is None:
            laparams = LAParams()

        with open_filename(pdf_file, 'rb') as fp:
            pdf_resource_manager = PDFResourceManager(caching=caching)
            idx = 1

            for page in PDFPage.get_pages(fp, page_numbers, maxpages=maxpages, caching=caching):
                with StringIO() as output_string:
                    device = TextConverter(pdf_file, output_string, codec=codec, laparams=laparams)
                    interpreter = PDFPageInterpreter(pdf_resource_manager, device)
                    interpreter.process_page(page)
                    yield idx, output_string.getvalue()
                    idx += 1


    def pdfminer_runner(self):
        pdf_path = "HSC26-Bangla1st-Paper.pdf"
        for count, page_text in self.iter_text_per_page(self.pdf_path):
            print(f"page# {count}: \n{page_text}")
            print()


    def pdf_plumber(self):
        loader = PDFPlumberLoader(self.pdf_path)
        pages = []

        for doc in loader.lazy_load():
            pages.append(doc)
            

    def pdf_miner(self):
        loader = PDFMinerLoader(self.pdf_path)
        pages = []

        for doc in loader.lazy_load():
            pages.append(doc)


class PDFLoaderAsImage:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        
    
    def pdf_page_2_image(self):
        images = convert_from_path(self.pdf_path, output_folder='pdf-images', fmt='jpeg')

        for i, image in enumerate(images):
            image.save(f"page_{i + 1}.jpg", 'JPEG')


class Image2Text:
    def __init__(self, images_folder_path="pdf-images", text_file_path="book-contents"):
        self.images_folder_path = images_folder_path
        self.text_files_path = text_file_path

    def extract_text_from_image(self):
        image_paths = os.listdir(self.images_folder_path)
        for i, image in enumerate(image_paths):
            image = os.path.join(self.images_folder_path, image)
            img = Image.open(image)
            text = pytesseract.image_to_string(img, lang="ben")
            self.write_to_textFile(text, i+1)


    def write_to_textFile(self, content, index):
        with open(f"{self.text_files_path}/page_{index}.txt", 'w') as file:
            file.write(f"{content}\n")


if __name__ == '__main__':
    pdf_path = "HSC26-Bangla1st-Paper.pdf"

    # image_pdf_loader = PDFLoaderAsImage(pdf_path)
    # image_pdf_loader.pdf_page_2_image("pdf-images/87b9c9f0-2bb0-4d08-8a15-f70a3882abf2-01.jpg")

    image2text = Image2Text()
    image2text.extract_text_from_image()
    