import streamlit as st
import sys
from dotenv import load_dotenv
from src.logger import logging
from src.exception import CustomException
from src.utils import get_pdf_text, get_text_chunks, get_vector_store, get_conversation_chain
from htmlTemplates import bot_template, css, user_template


def handle_userinput(user_question):
    logging("Generating answers for user question")
    
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)

    except Exception as e:
        raise CustomException(e, sys)

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with your PDF's", page_icon=":PDF:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your PDF's")
    user_question = st.text_input("Ask anything from the uploaded PDFs")
   
    if user_question:
        handle_userinput(user_question)

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
                #st.write(text_chunks)

                # create vector store
                vector_store = get_vector_store(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)




if __name__=='__main__':
    main()