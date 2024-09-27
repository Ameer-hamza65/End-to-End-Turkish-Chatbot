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

st.subheader("Tahlil: Yapay Zeka Asistanı", divider="rainbow", anchor=False)

# Initialize the Groq client with your API key
GROQ_API_KEY = st.secrets['GROQ_ACCESS_TOKEN']
client = Groq(api_key=GROQ_API_KEY)

# Initialize chat history, selected model, and feedback
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "feedback" not in st.session_state:
    st.session_state.feedback = []

# Define model details
models = {
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
    "llama-3.1-70b-versatile": {"name": "LLaMA-3.1-70b-versatile", "tokens": 8192, "developer": "Meta"},
    "llama-3.1-8b-instant": {"name": "LLaMA-3.1-8b-instant", "tokens": 8192, "developer": "Meta"}
}

# Sidebar for PDF uploader and model selection
with st.sidebar:
    uploaded_files = st.file_uploader("PDF belgeleri yükleyin", type="pdf", accept_multiple_files=True)

    model_option = st.selectbox(
        "Modeli seçin:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=4  # Default to mixtral
    )

    max_tokens_range = models[model_option]["tokens"]
    max_tokens = st.slider(
        "Maksimum Token:",
        min_value=512,
        max_value=max_tokens_range,
        value=min(32768, max_tokens_range),
        step=512,
        help=f"Modelin yanıtı için maksimum token (kelime) sayısını ayarlayın. Seçilen model için maksimum: {max_tokens_range}"
    )

# PDF upload functionality
def load_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_name = temp_file.name

            loader = PyPDFLoader(temp_file_name)
            documents.extend(loader.load())
            os.remove(temp_file_name)

        except Exception as e:
            st.error(f"{uploaded_file.name} yüklenirken bir hata oluştu: {e}")

    if not documents:
        st.error("Hiçbir belge yüklenmedi veya okunamadı.")
        return None
    
    return documents

def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=80)
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

if uploaded_files:
    documents = load_documents(uploaded_files)
    if documents:
        st.session_state.vectorstore = setup_vectorstore(documents)
        st.success("Belgeler yüklendi ve vektör deposu başarıyla oluşturuldu!")

# Display chat messages from history on app rerun
for index, message in enumerate(st.session_state.messages):
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍", key=f"thumbs_up_{index}"):
                    st.session_state.feedback.append((message["content"], "thumbs up"))
                    st.success("Teşekkür ederiz! Geri bildiriminiz alındı.")
            with col2:
                if st.button("👎", key=f"thumbs_down_{index}"):
                    st.session_state.feedback.append((message["content"], "thumbs down"))
                    st.success("Teşekkür ederiz! Geri bildiriminiz alındı.")

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def is_turkish(text):
    """A basic function to check if the response is in Turkish."""
    # A simple check for Turkish characters
    return any(char in text for char in 'şğüıöç')

if prompt := st.chat_input("Buraya isteğinizi girin..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    if st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever()
        retrieved_docs = retriever.get_relevant_documents(prompt)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Add explicit instruction to respond in Turkish
        messages = [{"role": "user", "content": f"{context}\n\nLütfen sadece Türkçe yanıt verin: {prompt}"}]
    else:
        messages = [{"role": "user", "content": f"Lütfen sadece Türkçe yanıt verin: {prompt}"}]

    messages = messages + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=messages,
            max_tokens=max_tokens,
            stream=True
        )

        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(e, icon="🚨")

    if isinstance(full_response, str):
        # Ensure the response is in Turkish; otherwise, raise an error
        if not is_turkish(full_response):
            st.error("Lütfen sadece Türkçe yanıt verin!", icon="🚨")
        else:
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        combined_response = "\n".join(str(item) for item in full_response)
        if not is_turkish(combined_response):
            st.error("Lütfen sadece Türkçe yanıt verin!", icon="🚨")
        else:
            st.session_state.messages.append({"role": "assistant", "content": combined_response})
