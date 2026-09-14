from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader('Introduction to Machine Learning.pdf')

documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
RecursiveCharacterTextSplitter()
chunks = splitter.split_documents(documents)

print('No.of chunks: ', len(chunks))
print('Chunk data:\n', chunks[0].page_content)
print('Chunk metadata:\n', chunks[0].metadata)