import os
import streamlit as st
import tempfile
from typing import Optional, List
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from together import Together


from dotenv import load_dotenv
load_dotenv()
ORG_PASSWORD = os.getenv("ORG_PASSWORD")

# Authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Pharma Regulatory Chatbot Login")
    pwd = st.text_input("Enter Access Password", type="password")
    if st.button("Login"):
        if pwd == ORG_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid password. Access denied.")
    st.stop()



# Load API Key
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    st.error("❌ TOGETHER_API_KEY not found in .env file!")
    st.stop()

# Custom Together LLM
class TogetherLLM(LLM):
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_tokens: int = 512
    temperature: float = 0.0
    api_key: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "together"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        client = Together(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=stop,
        )
        return response.choices[0].message.content.strip()

# Load vector store
def load_vectorstore(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        try:
            loader = PyPDFLoader(tmp_file_path)
            all_docs.extend(loader.load())
        finally:
            # Delete the temporary file after loading
            os.remove(tmp_file_path)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(all_docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # type: ignore
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

# Layout
st.set_page_config(page_title="Pharma Regulatory Chatbot", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: Arial !important;
    }
    .sidebar .sidebar-content {
        background-color: rgba(0, 0, 51, 0.9);
        color: white;
    }
    .chat-entry {
        border-radius: 10px;
        padding: 10px 15px;
        margin: 10px 0;
        word-break: break-word;
        font-size: 16px;
        white-space: pre-wrap;
    }
    .user-msg {
        background-color: #0d6efd;
        color: white;
        text-align: right;
    }
    .bot-msg {
        background-color: #333;
        color: #e0e0e0;
        text-align: left;
    }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    if st.button("New Chat"):
        st.session_state.clear()

    st.markdown("### Pharma Regulatory Assistant")
    st.markdown("Upload multiple pharma PDF documents to start.")

    if "chat_history" in st.session_state and st.session_state.chat_history:
        history_export = "\n\n".join([f"{s}: {m}" for s, m in st.session_state.chat_history])
        st.download_button("Export Chat", history_export, file_name="chat_history.txt")

# Title and upload
st.title("Pharma Regulatory Chatbot")
uploaded_files = st.file_uploader("Upload pharma regulatory PDFs:", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if "vectorstore" not in st.session_state:
        with st.spinner("Processing documents..."):
            st.session_state.vectorstore = load_vectorstore(uploaded_files)
            retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

            prompt_template = """
You are a precise and truthful pharma regulatory assistant AI. ONLY answer from the given context.
If the answer is NOT present in the context, reply exactly: 'Not found in context'.

Context:
{context}

Question: {question}
Answer:"""
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            llm = TogetherLLM(api_key=TOGETHER_API_KEY)
            st.session_state.qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt}
            )

    # Init chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat display container
    chat_container = st.container()

    # Chat form
    with st.form("query_form", clear_on_submit=True):
        query = st.text_input("Ask your question here...", placeholder="e.g. What are FDA rules for packaging?")
        submitted = st.form_submit_button("Ask")
        if submitted and query:
            answer = st.session_state.qa.run(query)
            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("Answer", answer))

    # Render chat (oldest at top, newest at bottom)
    with chat_container:
        for user, bot in zip(st.session_state.chat_history[::2], st.session_state.chat_history[1::2]):
            st.markdown(f'<div class="chat-entry user-msg">{user[1]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-entry bot-msg">{bot[1]}</div>', unsafe_allow_html=True)

        scroll_anchor = st.empty()
        scroll_anchor.markdown("<div id='scroll-to-latest'></div>", unsafe_allow_html=True)

    # Scroll to bottom using JS
    st.markdown("""
        <script>
            var elem = document.getElementById("scroll-to-latest");
            if (elem) {
                elem.scrollIntoView({behavior: "smooth"});
            }
        </script>
    """, unsafe_allow_html=True)

else:
    st.info("Upload at least one PDF to begin.")

    # Optionally allow user to query without PDFs
    st.markdown("---")
    st.markdown("General Pharma Regulatory Assistant")

    with st.form("general_query_form", clear_on_submit=True):
        general_query = st.text_input("Enter your general query:")
        submitted_general = st.form_submit_button("Submit")
        if submitted_general and general_query:
            general_llm = TogetherLLM(api_key=TOGETHER_API_KEY)
            response = general_llm(general_query)
            st.markdown(f'<div class="chat-entry user-msg">{general_query}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-entry bot-msg">{response}</div>', unsafe_allow_html=True)
