import streamlit as st
import tempfile
from ingestion import process_document
from rag import answer_rag
import hashlib

st.set_page_config(
    page_title="DocuLens",
    page_icon="📄",
    layout="wide"
)

import os
from dotenv import load_dotenv
load_dotenv()
if not os.getenv('GEMINI_API_KEY'):
    st.error("Google API key is not configured.")
    st.stop()

#Get the uploaded file as temp file from streamlit
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(uploaded_file.getvalue())

    return temp_file.name

def get_uploaded_file_id(uploaded_file):
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "temp_path" not in st.session_state:
    st.session_state.temp_path = None

if "file_id" not in st.session_state:
    st.session_state.file_id = None

if "vector_base" not in st.session_state:
    st.session_state.vector_base = None

st.title("📄 DocuLens")
st.caption("Ask questions about your documents using conversational RAG.")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    file_id = get_uploaded_file_id(uploaded_file)

    if st.session_state.get("file_id") != file_id:

        old_temp_path = st.session_state.temp_path

        if old_temp_path and os.path.exists(old_temp_path):
            os.remove(old_temp_path)

        st.session_state.file_id = file_id
        st.session_state.chat_history = []
        st.session_state.temp_path = save_uploaded_file(uploaded_file)

        with st.spinner("Processing document..."):
            st.session_state.vector_base = process_document(
                st.session_state.temp_path
            )

    st.write("File uploaded:", uploaded_file.name)

    temp_path = st.session_state.temp_path   
    vector_base = st.session_state.vector_base

    st.write('No.of chunks: ', vector_base._collection.count())

    with st.form("question_form"):
        query = st.text_input("Ask a question about your document")
        submitted = st.form_submit_button("Ask Question")

    for message in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(message["question"])

        with st.chat_message("assistant"):
            st.write(message["answer"])

        with st.expander("Sources"):
            for chunk in message["sources"]:
                page_num = chunk.metadata["page"]

                st.write(f"**Page {page_num}**")
                st.write(chunk.page_content)
    
    if submitted and query:

        with st.spinner('Thinking...'):
            response, results = answer_rag(vector_base, query, st.session_state.chat_history)

        st.session_state.chat_history.append(
            {
                'question' : query,
                'answer' : response,
                'sources' : results
            }
        )

        st.rerun()