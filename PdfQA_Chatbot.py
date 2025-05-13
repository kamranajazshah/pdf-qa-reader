import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import re
import os


load_dotenv()


st.set_page_config(page_title="PDF Q&A Reader", layout="wide")


embedding = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo")


def text_preprocess(text):
    text = re.sub(r'DOI:.*?\|\s*Page \d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_preprocess_pdf(pdf_path):
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    clean_texts = []
    for doc in documents:
        raw_text = doc.page_content
        clean_texts.append(text_preprocess(raw_text))
    return ' '.join(clean_texts)


st.title("📄 Ask Questions from PDF using LangChain")

pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

if pdf_file is not None:
    with st.spinner("Processing PDF..."):

        temp_pdf_path = f"./temp_{pdf_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())

        clean_text = load_preprocess_pdf(temp_pdf_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        splitted_texts = text_splitter.split_text(clean_text)

       
        vectorstore = FAISS.from_texts(splitted_texts, embedding=embedding)
        retriever = vectorstore.as_retriever()

        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("PDF processed. You can now ask questions!")

        user_query = st.text_input("Ask a question about the PDF:")
        if user_query:
            with st.spinner("Thinking..."):
                response = qa_chain.invoke(user_query)
                st.markdown("**Answer:**")
                st.write(response["result"])


        os.remove(temp_pdf_path)
