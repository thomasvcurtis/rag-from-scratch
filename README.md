# RAG from Scratch

A simple Retrieval-Augmented Generation (RAG) system built from scratch using OpenAI embeddings and ChromaDB for learning purposes.

## Overview

This project implements a basic RAG pipeline that:
1. Loads text documents from a data directory
2. Splits them into manageable chunks
3. Generates embeddings using OpenAI's embedding API
4. Stores the embeddings in ChromaDB for vector similarity search
5. Retrieves relevant context for user queries
6. Generates philosophically-inspired responses using OpenAI's chat completion API

## Features

- **Document Processing**: Automatically loads and processes `.txt` files from the data directory
- **Text Chunking**: Splits large documents into smaller chunks with configurable overlap
- **Vector Embeddings**: Uses OpenAI's `text-embedding-3-small` model for semantic search
- **Persistent Storage**: ChromaDB provides persistent vector storage
- **Contextual Responses**: Generates responses in the style of ancient philosophers (Socrates, Plato, Aristotle)

## Project Structure

```
.
├── .env                    # Environment variables (OpenAI API key)
├── basic-rag.py           # Main RAG implementation
├── requirements.txt       # Python dependencies
├── data/                  # Text documents directory
│   ├── discourses.txt     # Epictetus discourses
│   └── meditations.txt    # Marcus Aurelius meditations
└── chroma_storage/        # ChromaDB persistent storage
    └── ...               # Database files
```

## Setup

1. **Clone the repository** and navigate to the project directory

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Add your documents**:
   Place your `.txt` files in the `data/` directory

## Usage

Run the RAG system:

```bash
python basic-rag.py
```

The script will:
1. Load documents from the `data/` directory
2. Process and chunk the text
3. Generate embeddings and store them in ChromaDB
4. Execute an example query about Stoic philosophy
5. Return a philosophically-styled response

### Key Functions

- **[`load_documents_from_directory()`](basic-rag.py)**: Loads all `.txt` files from a directory
- **[`split_text()`](basic-rag.py)**: Splits text into chunks with configurable size and overlap
- **[`get_openai_embedding()`](basic-rag.py)**: Generates embeddings using OpenAI API
- **[`query_documents()`](basic-rag.py)**: Retrieves relevant document chunks for a query
- **[`generate_response()`](basic-rag.py)**: Generates contextual responses using retrieved chunks

## Configuration

You can modify these settings in [basic-rag.py](basic-rag.py):

- `chunk_size`: Size of text chunks (default: 1000 characters)
- `chunk_overlap`: Overlap between chunks (default: 20 characters)
- `n_results`: Number of relevant chunks to retrieve (default: 2)

## Example Query

The default example asks: *"What does Epictetus say our chief task in life is as Stoics"*

The system will:
1. Find relevant chunks from the stored documents
2. Generate a response in the style of ancient philosophers
3. Provide concise, wisdom-filled answers with rhetorical questions and analogies

## Dependencies

- `chromadb`: Vector database for storing and querying embeddings
- `openai`: OpenAI API client for embeddings and chat completions
- `python-dotenv`: Environment variable management

## Learning Notes

This implementation demonstrates:
- Basic RAG architecture and workflow
- Vector similarity search with embeddings
- Document chunking strategies
- Integration between different AI services
- Persistent vector storage
