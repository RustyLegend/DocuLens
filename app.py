import streamlit as st
import tempfile
from build_index import process_document
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

def create_context(results):
    return '\n\n'.join(f"Page: {x.metadata['page']}\n{x.page_content}" for x in results)

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

    query = st.text_input('Ask a question about your document: ')

    if query:
        retriever = vector_base.as_retriever(search_kwargs = {'k' : 5})
        results = retriever.invoke(query)
        context = create_context(results)

        load_dotenv()
        llm = GoogleGenerativeAI(
            model = 'gemini-3.6-flash'
        )

        prompt = ChatPromptTemplate.from_template(
        """
        Use only the information provided in the retrieved passages.
        Answer the question clearly and concisely.

        If the passages do not contain enough information to answer the
        question, say that the answer cannot be determined from the
        provided passages.

        Retrieved passages:
        {context}

        Question:
        {question}
            """
        )

        message = prompt.invoke({
            'context' : context,
            'question' : query
        })

        response = llm.invoke(message)

        st.write('##Answer')

        st.write(response)

        #st.write('Retrieved chunks: ')
        #for chunk in results:
        #    st.write(chunk.metadata['page'])
        #    st.write(chunk.page_content)