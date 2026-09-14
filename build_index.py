from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#Load document
loader = PyPDFLoader('Introduction to Machine Learning.pdf')
documents = loader.load()

#Create chunks of imported document
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunks = splitter.split_documents(documents)

#Embed created chunks and store them in chroma DB
embedding_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
vectorbase = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory='./chroma_db'
)

print('No.of chunks stored in DB: ', vectorbase._collection.count())