from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#Load document
def load_document(path):
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents

#Create chunks of imported document
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = splitter.split_documents(documents)
    return chunks

#Embed created chunks and store them in chroma DB
def create_vector_db(chunks, persist_dir):
    embedding_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    vectorbase = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    return vectorbase

documents = load_document('Introduction to Machine Learning.pdf')
print('No.of pages: ', len(documents))

chunks = split_documents(documents)
print('No.of chunks: ', len(chunks))

vectorbase = create_vector_db(chunks, './chroma_db')
print('No.of chunks added to vector DB: ', vectorbase._collection.count())