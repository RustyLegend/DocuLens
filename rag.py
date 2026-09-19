from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

def create_context(results):
    return '\n\n'.join(f"Page: {x.metadata['page']}\n{x.page_content}" for x in results)

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

load_dotenv()
llm = GoogleGenerativeAI(
    model = 'gemini-3.6-flash'
)

def answer_rag(vector_base, query):
    retriever = vector_base.as_retriever(search_kwargs = {'k' : 5})
    results = retriever.invoke(query)
    context = create_context(results)
    
    message = prompt.invoke({
        'context' : context,
        'question' : query
    })

    response = llm.invoke(message)

    return response, results