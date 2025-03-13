from pydantic import BaseModel, Field
from src.rag.file_loader import Loader
from src.rag.vectorstore import VectorDB
from src.rag.rag import RAG_Chain

class InputQA(BaseModel):
    """Input data model for the question answering API"""
    question: str = Field(..., title="Question to ask the model")

class OutputQA(BaseModel):
    """Output data model for the question answering API"""
    answer: str = Field(..., title="Answer from the model")

def build_rag_chain(llm, data_dir, data_type):
    """Build the RAG chain for the question answering API"""
    doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers=2)
    retriever = VectorDB(documents = doc_loaded).get_retriever()
    rag_chain = RAG_Chain(llm).get_chain(retriever)
    return rag_chain
