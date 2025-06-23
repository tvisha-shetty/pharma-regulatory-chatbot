import os
import tempfile #temporarily save the uploaded PDF for processing.
import streamlit as st #build the web interface.
from typing import Optional, List
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA #allows asking questions over pdf
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from together import Together

# Load API Key
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    st.error("TOGETHER_API_KEY not found in .env file!")
    st.stop()

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

@st.cache_resource(show_spinner=True)
def load_vectorstore(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        #uploadedd files are saved to temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

# Layout and theme
st.set_page_config(page_title="Pharma Regulatory Chatbot", layout="centered")

# Sidebar
with st.sidebar:
   #st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Pharmacy_symbol.png/800px-Pharmacy_symbol.png", width=180)
    st.markdown("**Pharma Regulatory Assistant**")
    st.markdown("Upload PDF guidelines, SOPs, or FDA documents.")
    if st.button("Export Chat"):
        chat_text = "\n\n".join([f"{s}: {m}" for s, m in st.session_state.get("chat_history", [])])
        st.download_button("Download", data=chat_text, file_name="chat_history.txt")

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #121212;
    color: #e0e0e0;
    font-family: Arial, sans-serif;
    max-width: 750px;
    margin: auto;
    padding: 2rem;
}
.stTextInput > div > input {
    background-color: #1e1e1e;
    color: #e0e0e0;
    border-radius: 6px;
    border: 1px solid #444;
    padding: 10px;
    font-size: 16px;
}
.stButton>button {
    background-color: #0d6efd;
    color: white;
    font-weight: 600;
    width: 100%;
    padding: 0.6rem;
    border-radius: 6px;
    border: none;
    cursor: pointer;
}
.stButton>button:hover {
    background-color: #0b5ed7;
}
.chat-entry {
    border-radius: 10px;
    padding: 15px 20px;
    margin: 10px 0;
    max-width: 80%;
    word-break: break-word;
    font-size: 16px;
    line-height: 1.5;
    white-space: pre-line;
}
.user-msg {
    background-color: #0d6efd;
    color: white;
    margin-left: auto;
    text-align: right;
}
.bot-msg {
    background-color: #333;
    color: #e0e0e0;
    margin-right: auto;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("Pharma Regulatory Chatbot")

#lets multiples pdfs , continues only if one file at least is uploaded
uploaded_files = st.file_uploader(
    "Upload pharma regulatory PDF files:",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files: #loads the cached functions and top 10 relevant chunks
    with st.spinner("Processing documents..."):
        vectorstore = load_vectorstore(uploaded_files)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        prompt_template = """
You are a precise and truthful pharma regulatory assistant AI. ONLY answer from the given context.
If the answer is NOT present in the context, reply exactly: 'Not found in context'.

Context:
{context}

Question: {question}
Answer:"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        #custom model class with access to model
        llm = TogetherLLM(api_key=TOGETHER_API_KEY)
        qa = RetrievalQA.from_chain_type(
            llm=llm, #use mistral model
            retriever=retriever,#use retriever to search in pdf
            chain_type="stuff", #stuff all the relevant chunks together into one big input
            chain_type_kwargs={"prompt": prompt}, #use the prompt template to guide the model
        )
#checks if chat has stated 
#and if has then creates empty list to store all the messages 
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    #loops through in reverse to show newest to oldest
    for sender, message in reversed(st.session_state.chat_history):
        css_class = "user-msg" if sender == "You" else "bot-msg"
        st.markdown(f'<div class="chat-entry {css_class}">{message.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
#you: blue bubble
#bot: grey bubble
   
   
    #false: text box was submitted and then the bots answer
    submitted = False
    answer = ""

    with st.form("chat_form", clear_on_submit=True):
        #after ask it clears 
        user_input = st.text_input("Type your pharma regulatory query here...", key="input", placeholder="e.g., What is the SOP for tablet packaging?")
        #stores question
        submitted = st.form_submit_button("Ask")

#if something is submitted
    if submitted and user_input:
        with st.spinner("Getting answer..."):
            answer = qa.run(user_input) #sends question to retreival qa system
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Answer", answer))
            st.rerun()  # refreshes page to see new question and answer
           
            

else:
    st.info("Please upload at least one PDF file to begin.")

st.markdown('</div>', unsafe_allow_html=True)
