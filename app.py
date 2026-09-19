import streamlit as st
import tempfile
from build_index import process_document
from rag import answer_rag

#Get the uploaded file as temp file from streamlit
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(uploaded_file.getvalue())

    return temp_file.name

st.title("Document Q&A")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    st.write("File uploaded:", uploaded_file.name)

    temp_path = save_uploaded_file(uploaded_file)

    vector_base = process_document(temp_path)

    st.write('No.of chunks: ', vector_base._collection.count())

    with st.form("question_form"):
        query = st.text_input("Ask a question about your document")
        submitted = st.form_submit_button("Ask Question")
    
    if submitted and query:

        response, results = answer_rag(vector_base, query)

        st.write('### Answer')
        st.write(response)

        st.write('### Sources')
        for chunk in results:

            page_num = chunk.metadata['page']

            with st.expander(f'Page: {page_num}'):
                st.write(chunk.page_content)