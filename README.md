# DocuLens

DocuLens is a conversational Retrieval-Augmented Generation (RAG) application for PDF documents. You upload a PDF, and then ask questions about it in a chat interface.

The point of the project is grounding. A general-purpose LLM answers from whatever it absorbed during training, which is useless when the question is about a specific document it has never seen. DocuLens instead searches the uploaded document for the passages most relevant to your question, hands those passages to the model, and asks it to answer from them. Every answer comes with the passages that produced it, so you can check the reasoning yourself.

Follow-up questions work too. If you ask "What is L1 regularization?" and then "What does it do to coefficients?", the second question is meaningless to a retriever on its own, so DocuLens rewrites it into a standalone question before searching.

Repository: https://github.com/RustyLegend/DocuLens

---

## Features

- PDF upload through the browser
- Semantic retrieval over the document using vector embeddings
- Conversational question answering with history
- Automatic query rewriting so follow-up questions retrieve correctly
- Maximum Marginal Relevance (MMR) retrieval, which trades a little relevance for diversity among the retrieved chunks
- Retrieved source passages displayed alongside every answer
- Persistent, per-document Chroma vector stores
- Re-uploading a document reuses its existing vector store instead of rebuilding it
- A generation prompt that instructs the model to decline rather than answer from general knowledge when the passages are insufficient
- Streamlit web interface

---

## How the pipeline works

The system has two halves: an ingestion path that runs once per document, and a query path that runs once per question.

### Ingestion

**Loading.** The uploaded PDF is written to a temporary file and read page by page with LangChain's `PyPDFLoader`. Each page becomes a document object carrying its text along with metadata such as the page number, which is what later lets the UI tell you where an answer came from.

**Chunking.** A whole page is too coarse a unit for retrieval, so the text is split with a recursive character splitter configured for chunks of roughly 1000 characters with a 200-character overlap between neighbours. The recursive splitter tries to break on paragraph boundaries first, then sentences, then words, so chunks tend to end at natural seams rather than mid-word. The overlap exists because a sentence that straddles a chunk boundary would otherwise be cut in half and lose its meaning in both halves; repeating a couple of hundred characters is a cheap insurance policy against that.

**Embedding.** Each chunk is converted into a dense vector using the `sentence-transformers/all-MiniLM-L6-v2` model through LangChain's HuggingFace embeddings wrapper. The model runs locally, so no document text is sent anywhere for this step. Chunks with similar meaning end up close together in vector space, which is what makes semantic search possible: the query "how do I penalise large weights?" can retrieve a passage about regularization even though they share almost no words.

**Storage.** The vectors and their source chunks go into a Chroma collection persisted to disk under `vector_stores/`, in a subdirectory named after the document's ID.

### Retrieval and generation

**Query rewriting.** Before anything is retrieved, the current question and the conversation so far are sent to Gemini with an instruction to produce a single self-contained question. A pronoun-heavy follow-up like "how is that different from L2?" becomes something a retriever can actually match against. If the question is already standalone, the rewrite is close to a no-op.

**MMR retrieval.** The rewritten question is embedded and used to search Chroma. The retriever is configured for Maximum Marginal Relevance with `fetch_k = 20` and `k = 5`: it pulls the 20 nearest chunks, then greedily picks 5 of them, penalising each candidate by how similar it already is to the ones already selected. Plain top-k similarity search has a failure mode where all five results are near-duplicates of the same paragraph, which wastes the context window and hides information the answer needs. MMR is the standard fix.

**Context construction.** The selected chunks are concatenated into a single context block, which is inserted into a prompt template along with the question.

**Answer generation.** That prompt goes to Gemini with three instructions: use only the supplied passages, answer clearly and concisely, and state that the answer cannot be determined if the passages do not contain enough information. The function returns both the generated answer and the retrieved document objects, so the interface can display them together.

---

## Conversational behaviour

Conversation state lives in Streamlit's session state as a list of entries, each holding the question, the generated answer, and the source documents retrieved for it. That single structure serves three purposes: it renders the chat transcript, it feeds the query-rewriting step, and it keeps each answer's sources attached to the answer rather than to the session as a whole.

Uploading a different document resets the conversation, since history from one document would only confuse retrieval over another.

---

## Document caching

Each uploaded file is hashed with MD5, and the resulting hex digest is used as the document's ID and as the name of its vector store directory. Two consequences follow. First, different documents get different IDs and therefore separate collections, so passages from one document can never leak into answers about another. Second, uploading the same file twice produces the same ID, so DocuLens can detect that the directory already exists and load the store instead of re-running the embedding pass, which is the slow part of ingestion.

MD5 is used here as a content fingerprint, not for security. Collisions are a non-issue for this purpose, though a non-cryptographic hash such as xxHash or a SHA-256 truncation would be equally valid choices.

---

## Grounding

The generation prompt explicitly restricts the model to the retrieved passages and tells it to say the answer cannot be determined when they fall short. This matters because the failure mode of a document Q&A tool is not a blank answer, it is a fluent answer assembled from the model's general knowledge that sounds like it came from your document.

This reduces unsupported answers. It does not eliminate them. No prompt can guarantee that an LLM will never step outside its context, which is exactly why the sources panel exists.

---

## Project structure

The repository is organised as three Python modules plus configuration.

`app.py` is the Streamlit layer. It handles the file uploader, writes the upload to a temporary file, detects when the user has switched documents, maintains chat history in session state, calls into the RAG pipeline, and renders answers together with an expandable sources section.

`ingestion.py` owns everything document-related: computing the file ID, checking whether a vector store already exists, loading and splitting the PDF, building embeddings, and creating or loading the Chroma store.

`rag.py` owns the query path: query rewriting, retrieval, context construction, prompt assembly, and the call to Gemini. It returns the answer along with the documents that were retrieved.

`requirements.txt` pins the dependencies. `.env` holds the Gemini API key locally and is not committed. `vector_stores/` holds generated indexes and is likewise excluded from version control via `.gitignore`.

---

## Technologies used

| Technology | Purpose |
| --- | --- |
| Python | Application language |
| Streamlit | Web interface |
| LangChain | RAG orchestration |
| Chroma (via langchain-chroma) | Vector database |
| PyPDF | PDF loading |
| Sentence Transformers | Embedding generation |
| all-MiniLM-L6-v2 | Embedding model |
| Google Gemini | Query rewriting and answer generation |
| python-dotenv | Environment variable management |

---

## Setup

Clone the repository and enter it:

```bash
git clone https://github.com/RustyLegend/DocuLens.git
cd DocuLens
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The first run will download the MiniLM embedding model, which takes a moment and needs a network connection. After that it is cached locally.

### API key

DocuLens needs a Google Gemini API key. Create a `.env` file in the project root containing a single line:

```
GEMINI_API_KEY=your_api_key_here
```

The application checks for this key at startup and reports a clear error if it is missing. Do not commit `.env`.

### Running

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in the terminal.

---

## Usage

Upload a PDF. DocuLens loads it, splits it into chunks, embeds them, and builds the vector store. This takes longest on the first upload of a given document; subsequent uploads of the same file are near-instant because the store is reused.

Ask a question. The question is rewritten if necessary, the five most relevant and mutually distinct chunks are retrieved, and Gemini answers from them.

Ask follow-ups freely. A sequence like "What is L1 regularization?", then "What does it do to coefficients?", then "How is that different from L2?" works, because each question is expanded against the conversation history before retrieval.

Expand the sources section under any answer to see the exact passages the model was given, along with their page numbers.

---

## Current limitations

- Only PDF uploads are supported.
- Each question costs two Gemini calls, one for rewriting and one for generation, so latency is roughly double a single-call system and depends on the API's response time.
- Vector stores are written to the local filesystem, which does not survive an ephemeral deployment container.
- Page numbers reflect whatever metadata the PDF loader produces, which can disagree with printed page numbers in documents with front matter.
- The application assumes a single user working locally or on a single-tenant deployment. There is no authentication or document management.
- Scanned PDFs without a text layer will not work, since no OCR step is present.

---

## Possible improvements

Support for DOCX and TXT inputs. Streaming responses so the answer appears as it is generated rather than all at once. Skipping the rewrite call when the question has no pronouns or ellipsis, which would halve latency for most questions. Retrieval evaluation, and then automated RAG metrics such as context precision and answer faithfulness, so changes to chunk size or `k` can be measured rather than guessed at. A hosted vector database. Authentication and per-user document management. Page-level navigation from a source passage back into the document. Deployment to a cloud platform.

---

## What this project covers

The project was built to understand the moving parts of a RAG system rather than to call a prebuilt chain and stop there. Working through it touches document loading, chunking and overlap, embeddings and semantic similarity, vector databases, similarity versus MMR retrieval, query rewriting for conversational retrieval, context construction, grounded generation, source attribution, persistent indexes, and session state management.

---

## License

Intended for educational and demonstration purposes. If you process documents with this application, make sure you have the rights to do so.

---

## Author

Rahul Gunturu — https://github.com/RustyLegend