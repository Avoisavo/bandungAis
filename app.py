import os
import openai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import streamlit as st

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to read PDFs and extract text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into smaller chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# Generate embeddings and create vector store
def get_vector_store(chunks):
    embeddings = []
    for chunk in chunks:
        response = openai.Embedding.create(
            input=[chunk],  # Input must be a list
            model="text-embedding-ada-002"
        )
        embeddings.append(response['data'][0]['embedding'])

    # Create and save FAISS vector store
    vector_store = FAISS.from_embeddings(embeddings, chunks)
    vector_store.save_local("faiss_index")

# Load conversational chain (using OpenAI GPT model)
def get_conversational_chain(question, context):
    prompt = f"""
    Context: {context}
    Question: {question}
    Answer:
    """
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Clear chat history
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question!"}]

# Handle user input and query vector store
def user_input(user_question):
    vector_store = FAISS.load_local("faiss_index")
    docs = vector_store.similarity_search(user_question, k=5)

    # Concatenate retrieved documents for context
    context = "\n".join([doc.page_content for doc in docs])
    response = get_conversational_chain(user_question, context)
    return response

# Streamlit app main function
def main():
    st.set_page_config(
        page_title="OpenAI PDF Chatbot",
        page_icon="🤖"
    )

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    st.title("Chat with PDF files using OpenAI 🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
