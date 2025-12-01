# ==== Basic Imports ====
import os
import tempfile  # Used to temporarily save uploaded files during processing
import streamlit as st  # Streamlit for the web interface

# ==== LangChain: Document Loading ====
from langchain_community.document_loaders import PyMuPDFLoader  # Extracts text and metadata from PDF files

# ==== LangChain: Chunking ====
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Breaks long documents into smaller chunks

# ==== LangChain: Vector Embeddings ====
import faiss  # Facebook AI Similarity Search — used for fast vector similarity search
from langchain_ollama import OllamaEmbeddings  # Embeddings from Ollama embedding models
from langchain_community.vectorstores import FAISS  # Wraps FAISS index to store and retrieve document vectors
from langchain_community.docstore.in_memory import InMemoryDocstore  # Keeps documents in memory for quick access

# ==== LangChain: RAG (Retrieval-Augmented Generation) ====
from langchain_ollama import ChatOllama  # Chat model from Ollama (like llama3.2)
from langchain_core.prompts import ChatPromptTemplate  # Helps build structured prompts
from langchain_core.output_parsers import StrOutputParser  # Parses LLM responses as plain strings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda  # Used in pipeline logic for RAG chain

# ==== Streamlit: App Initialization ====
# Initialize state variables across user interactions
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "current_answer" not in st.session_state:
    st.session_state.current_answer = ""

# ==== App UI Title ====
st.title("Chat with any PDF")
st.sidebar.title("Settings")

# ==== Sidebar: Model and Embedding Options ====
model_option = st.sidebar.selectbox(
    "Select LLM Model", ["llama3.2:1b"], index=0
)
embedding_option = st.sidebar.selectbox(
    "Select Embedding Model", ["nomic-embed-text"], index=0
)

# ==== Sidebar: Clear Chat Button ====
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.current_question = ""
    st.session_state.current_answer = ""

# ==== Function: Load, Process, and Chunk Uploaded Documents ====
def process_documents(uploaded_files):
    all_pages_content = []
    doc_locations = []

    for i, doc in enumerate(uploaded_files):
        # Save uploaded file to a temporary local path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(doc.read())
            tmp_path = tmp_file.name
            doc_locations.append(tmp_path)

        # Load and extract content from PDF
        loader = PyMuPDFLoader(tmp_path)
        pages = loader.load()

        # Add metadata (source name and page number)
        for page in pages:
            page.metadata["source"] = doc.name
            page.metadata["page"] = page.metadata.get("page", 0) + 1

        all_pages_content.extend(pages)

    # Chunk large texts into manageable parts for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_pages_content)

    return chunks, doc_locations

# ==== Function: Create a Vector Store from Chunks ====
def create_vector_store(chunks, embedding_option):
    with st.spinner("Creating vector store..."):
        # Generate vector embeddings for text chunks
        embeddings = OllamaEmbeddings(model=embedding_option, base_url='http://localhost:11434')
        vector_dimension = len(embeddings.embed_query("hello world"))

        # Create FAISS index with those embeddings
        index = faiss.IndexFlatL2(vector_dimension)
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),  # Stores the actual documents
            index_to_docstore_id={},  # Mapping from index to document ID
        )

        # Add document chunks to the FAISS index
        vector_store.add_documents(documents=chunks)
        return vector_store

# ==== Streamlit: File Upload Widget ====
uploaded_file = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# ==== Trigger: Process Uploaded Files ====
if uploaded_file:
    if st.button("Process Documents"):
        chunks, doc_locations = process_documents(uploaded_file)
        st.session_state.vector_store = create_vector_store(chunks, embedding_option)

        st.success(f"Processed {len(uploaded_file)} documents with {len(chunks)} total chunks")

        # Delete temporary files from disk
        for doc in doc_locations:
            try:
                os.remove(doc)
            except:
                pass

# ==== Function: Handle Q&A Interaction ====
def handle_question():
    question = st.session_state.question_input
    st.session_state.question_input = ""  # Reset input box

    if question and st.session_state.vector_store:
        # Store previous question/answer in history
        if st.session_state.current_question and st.session_state.current_answer:
            st.session_state.chat_history.append({
                "question": st.session_state.current_question,
                "answer": st.session_state.current_answer
            })

        st.session_state.current_question = question

        with st.spinner("Searching documents and generating answer..."):
            # Retrieve most relevant chunks based on similarity
            retrieved_chunks = st.session_state.vector_store.similarity_search(query=question, k=3)

            # Combine retrieved content into a single context string
            context = "\n\n".join([doc.page_content for doc in retrieved_chunks])

            # Create LLM (Ollama backend)
            model = ChatOllama(model=model_option, base_url='http://localhost:11434', temperature=0.7)

            # Define prompt template
            prompt = """
            You are a helpful assistant. Use the following extracted content from documents to answer the user's question.

            If the answer is not found in the content, respont with "This information is not present in the uploaded documents."

            Otherwise, answer in **exactly three bullet points** with a blank line between each bullet.
            
            DO NOT use outside knowledge. DO NOT guess. DO NOT add any information not explicitly in the context.

            Question:
            {question}

            Content:
            {context}

            Answer:
            """

            # Build RAG chain using the prompt and model
            prompt_template = ChatPromptTemplate.from_template(prompt)
            rag_chain = (
                {"context": RunnableLambda(lambda _: context), "question": RunnablePassthrough()}
                | prompt_template
                | model
                | StrOutputParser()
            )

            # Invoke model to get the answer
            response = rag_chain.invoke(question)
            st.session_state.current_answer = response

            # Show sources used in the response
            with st.expander("View Source Documents"):
                for doc in retrieved_chunks:
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "Unknown")
                    st.markdown(f"**{source} (Page {page})**")
                    st.markdown(doc.page_content)
                    st.divider()

# ==== UI: Input and Display Section ====
st.subheader("Ask questions about your documents")

# Input box for user's question
st.text_input("Enter your question:", key="question_input", on_change=handle_question)

# Display current Q&A
if st.session_state.current_question and st.session_state.current_answer:
    st.subheader(f"Question: {st.session_state.current_question}")
    st.markdown(st.session_state.current_answer)

# Display past chat history
if len(st.session_state.chat_history) > 0:
    st.subheader("Chat History")
    for message in st.session_state.chat_history:
        st.markdown(f"**Question:** {message['question']}")
        st.markdown(f"**Answer:** {message['answer']}")
        st.divider()

# Instructions when PDFs aren't loaded or processed yet
if not st.session_state.vector_store and not uploaded_file:
    st.info("Upload PDF documents to start chatting with them.")
elif not st.session_state.vector_store and uploaded_file:
    st.info("Click 'Process Documents' to analyze the uploaded PDFs.")