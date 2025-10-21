import os
import streamlit as st
import pickle
import time
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv

load_dotenv()  # Load environment variables (especially Hugging Face API key)

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_huggingface.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    # Load data using WebBaseLoader
    loader = WebBaseLoader(urls)
    main_placeholder.text("Data Loading... Started... ✅✅✅")
    data = loader.load()

    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitting... Started... ✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and store in FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building... ✅✅✅")
    time.sleep(2)

    # Save FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

            # Load Hugging Face model for QA
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",  # Use a text-generation model
                model_kwargs={"temperature": 0.5}
            )

            chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
            result = chain.run(query)

            st.header("Answer")
            st.write(result)