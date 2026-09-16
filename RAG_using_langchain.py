from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

#Fromat chunks into context for LLM
def create_context(results):
    return '\n\n'.join(x.page_content for x in results)

#Initialize sentence transformer model
model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

#Load vector DB
vectorbase = Chroma(
    persist_directory='./chroma_db',
    embedding_function= model
)

#Create a retriever object which returns top 5 results for a query
retriever = vectorbase.as_retriever(search_kwargs = {'k' : 5})

#Load API and initialize the model
load_dotenv()
llm = GoogleGenerativeAI(
    model='gemini-3.6-flash'
)

#Create prompt template
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

#Create langchain chain
rag_chain = (
    {
        'context' : retriever | create_context,
        'question' : RunnablePassthrough()
    }
    | prompt
    | llm
)

#Input query
query = input('Enter a query: ')

#Invoke RAG chain and print repsonse
response = rag_chain.invoke(query)
print(response)