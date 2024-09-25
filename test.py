import os
import tempfile
import streamlit as st
from typing import Generator, List
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Set the page configuration
st.set_page_config(page_icon="💬", layout="wide", page_title="Groq Goes Brrrrrrrr...")

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

icon("🏎️")

st.subheader("Groq Chat Streamlit App", divider="rainbow", anchor=False)

# Initialize the Groq client with your API key
client = Groq(api_key="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")

# Initialize chat history, selected model, and vector store
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Define model details
models = {
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
}

# Sidebar for model selection, PDF upload, and max_tokens slider
st.sidebar.header("Settings")

with st.sidebar:
    model_option = st.selectbox(
        "Choose a model:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=4  # Default to mixtral
    )

    max_tokens_range = models[model_option]["tokens"]

    max_tokens = st.slider(
        "Max Tokens:",
        min_value=512,  # Minimum value to allow some flexibility
        max_value=max_tokens_range,
        value=min(32768, max_tokens_range),
        step=512,
        help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
    )

    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

# PDF upload functionality
def load_documents(uploaded_files) -> List:
    """Load PDF documents and return a list of document objects."""
    documents = []
    for uploaded_file in uploaded_files:
        try:
            # Use a temporary file to store the uploaded PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_name = temp_file.name  # Save the temp file name for later use

            # Load the PDF from the temporary file
            loader = PyPDFLoader(temp_file_name)
            documents.extend(loader.load())
            os.remove(temp_file_name)

        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")

    if not documents:
        st.error("No documents were uploaded or could not be read.")
        return []
    
    return documents

@st.cache_data  # Cache the setup of the vector store
def setup_vectorstore(documents):
    """Set up a FAISS vector store from the provided documents."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=80)  # Adjust chunk size
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

# Load documents and setup vectorstore if files are uploaded
if uploaded_files:
    documents = load_documents(uploaded_files)
    if documents:
        if st.session_state.vectorstore is None:  # Only create a new vector store if it doesn't exist
            st.session_state.vectorstore = setup_vectorstore(documents)
            st.success("Documents loaded and vector store created successfully!")
        else:
            st.warning("Documents already loaded. You can query them now!")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

if prompt := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    # Check if vectorstore exists for document queries
    if st.session_state.vectorstore:
        # Use the vector store for document-based queries
        retriever = st.session_state.vectorstore.as_retriever()
        retrieved_docs = retriever.get_relevant_documents(prompt)  # Corrected this line
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Prepare messages for Groq API with a professional Turkish tone
        messages = [{"role": "user", "content": f"Bir profesyonel Türkçe ile cevap ver: {context}\n\n{prompt}"}]
    else:
        # Fallback to general knowledge with a professional Turkish tone
        messages = [{"role": "user", "content": f"Bir profesyonel Türkçe ile cevap ver: {prompt}"}]

    # Append previous messages to maintain context
    messages = messages + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=messages,
            max_tokens=max_tokens,
            stream=True
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(e, icon="🚨")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    else:
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})
