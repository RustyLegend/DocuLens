from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os

#Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

#Load stored vector DB
vectorbase = Chroma(
    persist_directory='./chroma_db',
    embedding_function=embedding_model
)

#LLM model initialization
load_dotenv()
llm = GoogleGenerativeAI(
    model = 'gemini-3.6-flash',
)

#Retrieve top 5 results for a query
retriever = vectorbase.as_retriever(search_kwargs = {'k' : 5})
query = input('Enter query: ')
results = retriever.invoke(query)

#Convert the top 5 results into context and build the prompt
prompt = ChatPromptTemplate.from_template(
    """
Use only the information provided in the retrieved passages.
Answer the question clearly and concisely.
Cite the page or page range where the information came from.

If the passages do not contain enough information to answer the
question, say that the answer cannot be determined from the
provided passages.

Retrieved passages:
{context}

Question:
{question}
    """
)
context = '\n\n'.join(x.page_content for x in results)
txt_prompt = prompt.invoke({
    'context' : context,
    'question' : query
})

#Generate response
response = llm.invoke(txt_prompt)
print(response)