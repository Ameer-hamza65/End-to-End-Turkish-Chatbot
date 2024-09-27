import os
import tempfile
import streamlit as st
from typing import Generator
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Set the page configuration
st.set_page_config(page_icon="💬", layout="wide", page_title="Tahlil: Yapay Zeka Asistanı")

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

# Initialize the Groq client with your API key
client = Groq(api_key='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')

# User authentication
def authenticate_user():
    if "username" not in st.session_state:
        st.session_state.username = st.text_input("Kullanıcı Adı")
        st.session_state.password = st.text_input("Şifre", type="password")
        if st.button("Giriş Yap"):
            if st.session_state.username == "admin" and st.session_state.password == "admin_pass":
                st.session_state.user_role = "admin"
                st.success("Admin olarak giriş yapıldı.")
            elif st.session_state.username == "student":
                st.session_state.user_role = "student"
                st.success("Öğrenci olarak giriş yapıldı.")
            else:
                st.error("Geçersiz kullanıcı adı veya şifre.")

# Authentication page
authenticate_user()

# Admin functionality
if st.session_state.get("user_role") == "admin":
    st.subheader("Admin Panel")
    
    uploaded_files = st.file_uploader("PDF belgeleri yükleyin", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_name = temp_file.name
            loader = PyPDFLoader(temp_file_name)
            documents.extend(loader.load())
            os.remove(temp_file_name)
        if documents:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=80)
            doc_chunks = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(doc_chunks, embeddings)
            st.session_state.vectorstore = vectorstore
            st.success("Belgeler yüklendi ve vektör deposu başarıyla oluşturuldu!")

# Student functionality
if st.session_state.get("user_role") == "student":
    st.subheader("Öğrenci Paneli")
    
    if "vectorstore" in st.session_state:
        prompt = st.text_input("Buraya isteğinizi girin...")
        if prompt:
            retriever = st.session_state.vectorstore.as_retriever()
            retrieved_docs = retriever.get_relevant_documents(prompt)
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            response = f"İşte aradığınız bilgi:\n{context}"
            st.write(response)
    else:
        st.warning("Admin belgeleri henüz yüklemedi.")

