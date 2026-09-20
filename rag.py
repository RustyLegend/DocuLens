from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

def create_context(results):
    return '\n\n'.join(f"Page: {x.metadata['page']}\n{x.page_content}" for x in results)

def rewrite_query(query, chat_history):
    history = '\n\n'.join(f'User: {message['question']}\nAI: {message['answer']}' for message in chat_history)

    rewrite_prompt = ChatPromptTemplate.from_template(
            """
    Given the conversation history and the user's latest question,
    rewrite the latest question into a standalone question that can
    be understood without the conversation history.

    If the latest question is already standalone, return it unchanged.

    Do not answer the question.

    Conversation history:
    {history}

    Latest question:
    {question}

    Standalone question:
    """
        )

    prompt = rewrite_prompt.invoke({
        'history' : history,
        'question' : query
    })

    response = llm.invoke(prompt)

    return response

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

def answer_rag(vector_base, query, chat_history):
    retriever = vector_base.as_retriever(
        search_type = 'mmr',
        search_kwargs = {
            'k' : 5 ,
            'fetch_k' : 20
        }
    )

    rewritten_query = rewrite_query(query, chat_history)

    results = retriever.invoke(rewritten_query)
    context = create_context(results)
    
    message = prompt.invoke({
        'context' : context,
        'question' : rewritten_query
    })

    response = llm.invoke(message)

    return response, results