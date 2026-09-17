from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from build_index import get_file_id, load_vector_db

#Fromat chunks into context for LLM
def create_context(results):
    return '\n\n'.join(f"Page: {x.metadata['page']}\n{x.page_content}" for x in results)

#Get DB object from file ID
file_path = 'Introduction to Machine Learning.pdf'
file_id = get_file_id(file_path)
vector_base = load_vector_db(f'./vector_stores/{file_id}')

#Create a retriever object which returns top 5 results for a query
retriever = vector_base.as_retriever(search_kwargs = {'k' : 5})

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

If the passages do not contain enough information to answer the
question, say that the answer cannot be determined from the
provided passages.

Retrieved passages:
{context}

Question:
{question}
    """
)

#Input query
query = input('Enter a query: ')

#Under Construction
#------------------------------------

results = retriever.invoke(query)
context = create_context(results)

message = prompt.invoke(
    {
        'context' : context,
        'question' : query
    }
)

response = llm.invoke(message)
print('\n\nAnswer:')
print(response)

print('\nSources:')
sources = set()
for chunk in results:
    sources.add(chunk.metadata['page'])

for page in sorted(sources):
    print('Page: ', page)
#------------------------------------

#Invoke RAG chain and print repsonse
#response = rag_chain.invoke(query)
#print(response)