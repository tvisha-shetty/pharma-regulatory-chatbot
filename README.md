An intelligent, secure chatbot designed for pharmaceutical professionals to instantly query SOPs, cGMP policies, and FDA regulations — directly from uploaded documents.

Built using LangChain, Mixtral-8x7B-Instruct-v0.1 via Together API, FAISS, BM25Retriever, and Streamlit, this chatbot transforms regulatory document search into a natural, interactive conversation.

Key Features:
Natural Language Questioning: Ask regulatory questions as you would to a colleague — no keywords or syntax needed.

Multi-Document Upload: Seamlessly upload and parse multiple PDF or DOCX files with intelligent chunking.

Strict Contextual Answers: Uses RAG + prompt engineering to avoid hallucinations. Returns “Not found in context” when unsupported.

Hybrid Retrieval: Combines FAISS (semantic) and BM25 (keyword) search for highly relevant results.

Conversational Memory: Supports multi-turn conversations with ConversationBufferMemory and ConversationalRetrievalChain.

Security-First Design: Single-password protected access with .env integration. Ideal for VPN/firewalled environments — no user database required.

Modern Streamlit UI:

Auto-scroll to latest message

Chat bubbles (bot vs user)

Sidebar with “New Chat” and “Export Chat”

Fallback general query option if no PDF is uploaded
