from typing import Union
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class VectorDB:
    """
    This class is used to build a vector database.
    """
    def __init__(self,
                 documents = None,
                 vector_db: Union[Chroma, FAISS] = Chroma,
                 embedding = HuggingFaceEmbeddings(),
                 ) -> None:
        self.vector_db = vector_db
        self.embedding = embedding
        self.db = self._build_db(documents)

    def _build_db(self, documents):
        """
        This function builds the vector database.
        """
        db = self.vector_db.from_documents(documents=documents, embedding=self.embedding)
        return db

    def get_retriever(self, search_type: str = "similarity", search_kwargs: dict = {"k": 10}):
        """
        This function returns a retriever object.
        """
        retriever = self.db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        return retriever
