import streamlit as st
from dotenv import load_dotenv
from src.logger import logging
from src.exception import CustomException
from src.utils import get_pdf_text, get_text_chunks, get_vector_store


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your PDF's", page_icon=":PDF:")

    st.header("Chat with your PDF's")
    st.text_input("Ask anything from the uploaded PDF")

    with st.sidebar:
        st.subheader("PDF upload")
        pdf_docs = st.file_uploader("Upload your PDF here", accept_multiple_files=True)
        if st.button("Process PDF"):
            with st.spinner("Processing"):
                # get PDF raw text
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text)

                # get text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # create vector store
                vector_store = get_vector_store(text_chunks)









if __name__=='__main__':
    main()