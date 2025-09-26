# AI Legal/Policy Agent — Interactive Dashboard (Prototype)

This prototype can read legal documents (PDF/Word), perform semantic searches, answer policy questions, classify simple impacts, and generate reports. It's suitable for demos to regional apparatus organizations or leaders.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Run the API first, make sure you in outside the app folder
uvicorn app.api:app --reload

# RUn the streamlit app
streamlit run app/app.py
```

## Notes
- Supported formats: .pdf / .docx.
- The embedding model will be downloaded on the first run (requires an internet connection).
- The FAISS index and ML model are stored in the `storage/` folder.

## Future Development
- Extraction of legal structures (articles/clauses) + document metadata.
- Cross-validation & hyperparameter tuning for the classification model.
- Recommendation reasoning + impact scoring (qualitative → quantitative).
- Export reports to PDF (currently markdown/HTML).
- Role-based access (Admin vs. Head of Regional Apparatus Organization).

## Workflow Explanation

### Upload & Indexing
1.  **Upload Document**: The user uploads a PDF/DOCX file from the dashboard.
2.  **Save to Folder**: The file is securely saved in the `data/` directory using `pathlib`.
3.  **Ingest & Chunking**: The text is extracted and split into smaller chunks.
4.  **Embedding**: The chunks are converted into vector embeddings (using Hugging Face / Ollama).
5.  **Vector DB (FAISS)**: The embeddings and metadata are stored.

### Query & Answering
1.  **User Query**: The user enters a question on the dashboard.
2.  **Query Embedding**: The question is converted into an embedding using the same model.
3.  **Semantic Search**: FAISS finds the most relevant Top-K contexts.
4.  **QA Agent**:
    *   **Mode 1: Extractive Answer** (without LLM, free).
    *   **Mode 2: Generative Answer** (LLM: OpenAI / Gemini / Ollama).
5.  **Display Answer**: The answer is displayed on the dashboard, with references to the source document.

## Version 1 — With LLM (RAG + Generative Answer)

This flowchart illustrates the complete process, from document upload to generating an answer using a Large Language Model (LLM). This approach, known as Retrieval-Augmented Generation (RAG), combines semantic search with the generative capabilities of an LLM to provide comprehensive answers.

```mermaid
flowchart TD
    subgraph Upload_&_Indexing
        A[Upload Document (PDF/DOCX) - Streamlit]
        B[Save to data/ folder (pathlib)]
        C[Ingest & Chunking (app/nlp/ingest.py)]
        D[Embedding (HF/Ollama)]
        E[(FAISS Index + Metadata) storage/]
    end

    subgraph Query_&_Answer
        F[User Query (Streamlit input)]
        G[Query Embedding (same model)]
        H[Semantic Search Top-K contexts]
        I[QA Agent (app/agent/qa_agent.py)\n→ Prompt = Query + Contexts]
        J[LLM Generation (OpenAI/Gemini/Ollama)]
        K[Answer + References (Dashboard)]
        L[[Download Report]]
    end

    A-->B-->C-->D-->E
    F-->G-->H-->I-->J-->K-->L
    E-->H
```

## Version 2 — Without LLM (Pure Extractive)

This flowchart shows a simplified, cost-effective version that operates without a Large Language Model (LLM). Instead of generating new text, this approach extracts and summarizes the most relevant information directly from the source documents. It's a pure extractive method that is faster and free of LLM-related costs.

```mermaid
flowchart TD
    subgraph Upload_&_Indexing
        A[Upload Document]
        B[Save to data/]
        C[Ingest & Chunking]
        D[Embedding]
        E[(FAISS Index)]
    end

    subgraph Query_&_Answer
        F[User Query]
        G[Query Embedding]
        H[Semantic Search Top-K]
        I[Extractive Answer\n(summarize context + quote)]
        J[Result + Source (Dashboard)]
        K[[Download Report]]
    end

    A-->B-->C-->D-->E
    F-->G-->H-->I-->J-->K
    E-->H
```

## app/ Directory Structure

The `app/` directory contains the main source code for this AI Legal/Policy Agent dashboard:

- `agent/` — Contains the QA (Question Answering) agent module responsible for context retrieval and answering user queries.
- `nlp/` — Natural Language Processing modules for document extraction, chunking, and embedding.
- `models/` — Impact classification models and related files.
- `utils/` — Utility functions such as report generation.
- `storage/` — Stores models, FAISS index, metadata, and uploaded documents.
- `app.py` — The entry point for the interactive dashboard application built with Streamlit.
- `backend/` — Contains the API code for handling document uploads, queries, and other backend operations.

Each folder plays a specific role in the workflow: from document upload, text processing, semantic search, to presenting answers and reports to users.

