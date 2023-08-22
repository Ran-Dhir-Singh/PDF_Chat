from src.logger import logging
from src.exception import CustomException
import sys
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

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
        logging.info("Created Text chunks")
        return chunks


    except Exception as e:
        raise CustomException(e, sys)
    


def get_vector_store(text_chunks):
    logging.info("Creting vector store")
    try:
        embeddings = OpenAIEmbeddings()
        #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        vector_Store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        logging.info("Created vector store")
        return vector_Store



    except Exception as e:
        raise CustomException(e, sys)
    


def get_conversation_chain(vector_store):
    logging.info("Creating conversation chain")

    try:
        llm = ChatOpenAI()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )
        logging("Created Conversation chain")
        return conversation_chain

    
    
    except Exception as e:
        raise CustomException(e, sys)
    




