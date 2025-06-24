import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os
import fitz  # PyMuPDF
import tempfile

# Set your OpenAI API key securely
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

st.set_page_config(page_title="RAG on PDF", page_icon="📄")
st.title("📄 RAG Over PDFs (Ask Questions on Uploaded Document)")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Extract and chunk text using PyMuPDF
    def extract_pdf_text(path):
        doc = fitz.open(path)
        return "\n".join([page.get_text() for page in doc])

    with st.spinner("📚 Reading and indexing your PDF..."):
        pdf_text = extract_pdf_text(pdf_path)
        with open("temp_text.txt", "w", encoding="utf-8") as f:
            f.write(pdf_text)
        
        documents = SimpleDirectoryReader(input_files=["temp_text.txt"]).load_data()
        llm = OpenAI(model="gpt-3.5-turbo")
        embed_model = OpenAIEmbedding()
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        query_engine = index.as_query_engine()

    question = st.text_input("💬 Ask a question about the PDF")

    if st.button("Get Answer") and question:
        with st.spinner("🤖 Thinking..."):
            response = query_engine.query(question)
            st.success("✅ Answer:")
            st.write(response.response)
