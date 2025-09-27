import streamlit as st
import fitz  # PyMuPDF for PDF reading
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# ---- PDF LOADING ----
def load_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ---- EMBEDDINGS & VECTOR DB ----
def create_vector_db(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(chunks, embeddings)
    return vectordb

# ---- OLLAMA QUERY ----
def ask_ollama(query, context):
    prompt = f"Answer the question based only on the following context:\n\n{context}\n\nQuestion: {query}"
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# ---- STREAMLIT UI ----
st.title("Conclubot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing document..."):
        text = load_pdf(uploaded_file)
        vectordb = create_vector_db(text)
    st.success("Document processed! Ask your questions below:")

    query = st.text_input("Enter your question:")
    if query:
        docs = vectordb.similarity_search(query, k=3)
        context = " ".join([d.page_content for d in docs])
        answer = ask_ollama(query, context)
        st.write("### Answer:")
        st.write(answer)
