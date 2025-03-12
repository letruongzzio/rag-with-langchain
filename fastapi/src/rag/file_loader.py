from typing import Union, List, Literal
import glob # This function is used to return all files matching a specified pattern
from tqdm import tqdm
import multiprocessing
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def remove_non_utf8_characters(text):
    """
    This function removes non-utf8 characters from the text.
    """
    return ''.join(char for char in text if ord(char) < 128) # ord() function returns the Unicode code point for the given character

def load_pdf(pdf_file):
    """
    This function loads the pdf file and removes non-utf8 characters from the text.
    """
    docs = PyPDFLoader(pdf_file, extract_images=True).load()
    for doc in docs:
        doc.page_content = remove_non_utf8_characters(doc.page_content)
    return docs

def load_html(html_file):
    """"
    This function loads the html file and removes non-utf8 characters from the text.
    """
    docs = BSHTMLLoader(html_file).load()
    for doc in docs:
        doc.page_content = remove_non_utf8_characters(doc.page_content)
    return docs

def get_num_cpu():
    """
    This function returns the number of CPUs.
    """
    return multiprocessing.cpu_count()

class BaseLoader:
    """
    This class is the base class for the loaders.
    """
    def __init__(self) -> None:
        self.num_processes = get_num_cpu()

    def __call__(self, files: List[str], **kwargs):
        pass

class PDFLoader(BaseLoader):
    """
    This class is used to load PDF files.
    """
    def __call__(self, pdf_files: List[str], **kwargs):
        num_processes = min(self.num_processes, kwargs["workers"])
        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded = []
            total_files = len(pdf_files)
            with tqdm(total=total_files, desc="Loading PDFs", unit="file") as pbar:
                # imap_unordered() function returns an iterator that returns the results of the function as they complete
                for result in pool.imap_unordered(load_pdf, pdf_files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        return doc_loaded
    
class HTMLLoader(BaseLoader):
    """
    This class is used to load HTML files.
    """
    def __call__(self, html_files: List[str], **kwargs):
        num_processes = min(self.num_processes, kwargs["workers"])
        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded = []
            total_files = len(html_files)
            with tqdm(total=total_files, desc="Loading HTMLs", unit="file") as pbar:
                for result in pool.imap_unordered(load_html, html_files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        return doc_loaded
    
class TextSplitter:
    """
    This class is used to split the text.
    """
    def __init__(self,
                 separators: Union[List[str], None] = None,
                 chunk_size: int = 300,
                 chunk_overlap: int = 0,
                 ) -> None:
        if separators is None:
            separators = ['\n\n', '\n', ' ', '']
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    def __call__(self, documents):
        return self.splitter.split_documents(documents)

class Loader:
    """
    This class is used to load the files."

    Args:
    file_type: str
        The type of file to load. It can be either pdf or html.
    split_kwargs: dict
        The keyword arguments for the TextSplitter class.
    """
    def __init__(self, 
                 file_type: str = Literal["pdf", "html"],
                 split_kwargs: dict = None
                 ) -> None:
        if split_kwargs is None:
            split_kwargs = {
                "chunk_size": 300,
                "chunk_overlap": 0
            }
        assert file_type in ["pdf", "html"], "file_type must be either pdf or html"
        self.file_type = file_type
        if file_type == "pdf":
            self.doc_loader = PDFLoader()
        elif file_type == "html":
            self.doc_loader = HTMLLoader()
        else:
            raise ValueError("file_type must be either pdf or html")

        self.doc_spltter = TextSplitter(**split_kwargs)

    def load(self, pdf_files: Union[str, List[str]], workers: int = 4):
        """
        This function loads the files and splits the text.
        """
        # If the pdf_files is a string, then convert it to a list.
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        doc_loaded = self.doc_loader(pdf_files, workers=workers)
        doc_split = self.doc_spltter(doc_loaded)
        return doc_split

    def load_dir(self, dir_path: str, workers: int = 4):
        """
        This function loads the files in the directory and splits the text.

        Args:
        dir_path: str
            The path to the directory containing the files.
        workers: int
            The number of workers to use for loading the files.

        Returns:
        doc_split: List[Document]
            The list of the documents with the text split.
        """
        if self.file_type == "pdf":
            files = glob.glob(f"{dir_path}/*.pdf")
            assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
        else:
            files = glob.glob(f"{dir_path}/*.html")
            assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
        return self.load(files, workers=workers)
