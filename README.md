🚀 Project Launch: Chat with Any PDF using LangChain, FAISS, Streamlit & LLaMA 3.2

I recently built a full-stack AI-powered PDF chatbot that allows users to upload multiple PDFs and ask natural language questions about their contents — all within an interactive Streamlit app.

🔍 Project Overview:
This project enables Retrieval-Augmented Generation (RAG) over PDF documents. Whether it's contracts, research papers, invoices, or manuals — users can ask questions and get accurate, cited responses from the uploaded documents in real time.

💡 Key Features:
📤 Upload multiple PDF files at once
🧠 Automatically processes and chunks documents into manageable text segments
🔎 Uses FAISS to create a vector store for fast similarity-based retrieval
🤖 Leverages Ollama’s LLaMA 3.2 LLM to generate responses using LangChain
🧩 Integrates LangChain RAG components for context-aware question answering
🧾 Displays source documents and page numbers for transparency
💬 Maintains chat history for ongoing multi-turn conversations

🧰 Tech Stack:
LangChain: Document loaders, text splitting, and chaining logic
FAISS: Fast and efficient vector similarity search
Ollama (LLaMA 3.2): Local language model for answering queries
Streamlit: UI for seamless interaction and chat interface
PyMuPDF: For reading and parsing PDF content

🌟 What I Learned:
Building end-to-end pipelines using LangChain Runnables for flexible chaining
Working with local LLMs (LLaMA 3.2) through Ollama — enabling private, offline inference
Handling multi-PDF ingestion, metadata tagging, and chunk-based retrieval
Creating a responsive and user-friendly app with Streamlit’s session state management
