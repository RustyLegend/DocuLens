# Machine Learning Textbook RAG with LangChain

A Retrieval-Augmented Generation (RAG) application built with LangChain, using *Introduction to Machine Learning with Python* by Andreas C. Müller and Sarah Guido as the knowledge source.

The application lets you ask questions about the textbook and get answers grounded in the passages actually retrieved from the book, rather than answers generated purely from the model's own knowledge.

## Features

- PDF document loading
- Automatic document chunking
- Semantic text embeddings
- Persistent ChromaDB vector database
- Similarity-based document retrieval, returning the top 5 relevant passages
- Context-aware prompting
- Answer generation powered by Google Gemini
- Page-based citations in generated answers
- Refusal when the retrieved passages don't contain enough information to answer

## Project Structure

```
rag-langchain/
│
├── Introduction to Machine Learning.pdf
├── build_index.py
├── RAG_using_langchain.py
├── requirements.txt
└── .gitignore
```

## Setup

### 1. Clone the Repository

Clone the repository and move into the project directory.

```bash
git clone https://github.com/RustyLegend/rag-v2.0-langchain
cd rag-v2.0-langchain
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the Gemini API key

Create a `.env` file in the project directory and add your Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

### 4. Build the vector database

The pre-built ChromaDB database isn't included in the repository, so it needs to be generated locally. Run:

```bash
python build_index.py
```

This creates the `chroma_db/` directory containing the vector index.

### 5. Run the RAG application

Once the vector database exists, run:

```bash
python RAG_using_langchain.py
```

Enter a question about the textbook when prompted.

## How It Works

```
Question -> Semantic Retrieval -> Top 5 Relevant Passages -> Context -> Gemini -> Grounded Answer
```

The application retrieves relevant passages from the textbook and provides them to Gemini as context for generating the answer, rather than letting the model answer from its own training data alone.

## Technologies

- Python
- LangChain
- ChromaDB
- Hugging Face Sentence Transformers
- Google Gemini
- PyPDF

## Embedding Model

The project uses `sentence-transformers/all-MiniLM-L6-v2`, which generates 384-dimensional embeddings for semantic search.

## Vector Database

ChromaDB is used as the persistent vector database. It's generated locally by `build_index.py` and is intentionally excluded from the repository, since it can always be rebuilt from the source PDF.

## Textbook

*Introduction to Machine Learning with Python* by Andreas C. Müller and Sarah Guido is the knowledge source for this project. The textbook is copyrighted by its respective rights holders and is included here for educational and demonstration purposes only. Redistribution of copyrighted material should only be done where legally permitted.

## Disclaimer

This project is intended for educational purposes and demonstrates the use of Retrieval-Augmented Generation with LangChain. Generated answers should be verified against the original textbook.