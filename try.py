from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Initialize BERT embeddings and QA pipeline
bert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Pretrained BERT model for embeddings
qa_pipeline = pipeline("question-answering", model="deepset/bert-base-cased-squad2")

# Read all PDF files and return text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# Get embeddings for each chunk and store in FAISS
def get_vector_store(chunks):
    # Generate embeddings for each chunk
    embeddings = [bert_model.encode(chunk) for chunk in chunks]

    # Pair chunks with their embeddings
    text_embedding_pairs = list(zip(chunks, embeddings))
    
    # Use FAISS to store the embeddings
    vector_store = FAISS.from_embeddings(text_embedding_pairs)
    vector_store.save_local("faiss_index")

# Retrieve similar documents and generate QA response
def user_input(user_question):
    vector_store = FAISS.load_local("faiss_index")
    docs = vector_store.similarity_search(user_question)

    # Use the BERT QA pipeline to answer the question from the most relevant document
    if docs:
        context = docs[0].page_content  # Take the most relevant document
        response = qa_pipeline(question=user_question, context=context)
        return response['answer']
    else:
        return "Answer is not available in the context."

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="BERT PDF Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("No text could be extracted from the uploaded PDFs. Please try again with a different file.")
                    return
                text_chunks = get_text_chunks(raw_text)
                if not text_chunks:
                    st.error("No valid text chunks were created. Please check the PDF formatting.")
                    return
                get_vector_store(text_chunks)
                st.success("Processing complete! You can now ask questions.")

    # Main content area for displaying chat messages
    st.title("Chat with PDF files using BERT🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=lambda: st.session_state.clear())

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Display chat messages and bot response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(prompt)
                    st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
