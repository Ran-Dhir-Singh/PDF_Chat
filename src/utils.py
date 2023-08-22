from src.logger import logging
from src.exception import CustomException
import sys
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

def get_pdf_text(pdf_docs):
    logging.info("Reading pdf files")
    try:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()

        return text
    
    except Exception as e:
        raise CustomException(e, sys)
    

def get_text_chunks(raw_texts):
    logging.info("Creating text chunks")

    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(raw_texts)

        return chunks


    except Exception as e:
        raise CustomException(e, sys)
    


def get_vector_store(text_chunks):
    logging.info("Creting vector store")
    try:
        #embeddings = OpenAIEmbeddings()
        
        vector_Store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vector_Store



    except Exception as e:
        raise CustomException(e, sys)
