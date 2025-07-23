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


if __name__ == '__main__':
    pdf_path = "HSC26-Bangla1st-Paper.pdf"
    image_pdf_loader = PDFLoaderAsImage(pdf_path)
    image_pdf_loader.pdf_page_2_image()

