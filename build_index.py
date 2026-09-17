from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import hashlib
import os

#Load document
def load_document(path):
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents

#Generate unique file id for files with different contents
def get_file_id(path):

    with open(path, 'rb') as file:
        data = file.read()

    file_id = hashlib.md5(data).hexdigest()

    return file_id

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

#Load vector DB
def load_vector_db(path):

    #Initialize sentence transformer model
    model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

    vectorbase = Chroma(
        persist_directory=path,
        embedding_function= model
    )

    return vectorbase

def process_document(doc_path):
    file_id = get_file_id(doc_path)
    persist_dir = f'./vector_stores/{file_id}'

    if os.path.exists(persist_dir):
        print('Document is already pre-processed. Loading it....')
        vector_base = load_vector_db(persist_dir)
        return vector_base

    print('Processing new document....')
    documents = load_document(doc_path)
    chunks = split_documents(documents)
    vector_base = create_vector_db(chunks, persist_dir)

    return vector_base

if __name__ == "__main__":

    path = 'Introduction to Machine Learning.pdf'

    vectorbase = process_document(path)

    print(
        'No.of chunks added to vector DB:',
        vectorbase._collection.count()
    )