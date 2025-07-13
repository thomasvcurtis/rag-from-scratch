# RAG from Scratch

A comprehensive Retrieval-Augmented Generation (RAG) system built from scratch with multiple implementation approaches for learning purposes.

## Overview

This project implements two RAG pipelines:

### Basic RAG (`basic-rag.py`)
A foundational RAG implementation that:
1. Loads text documents from a data directory
2. Splits them into manageable chunks
3. Generates embeddings using OpenAI's embedding API
4. Stores the embeddings in ChromaDB for vector similarity search
5. Retrieves relevant context for user queries
6. Generates philosophically-inspired responses using OpenAI's chat completion API

### Query Expansion RAG (`query-expansion-rag.py`)
An advanced RAG implementation featuring:
1. PDF document processing (Pro Git book)
2. Advanced text chunking with LangChain's recursive and token-based splitters
3. Query expansion using hypothetical answer generation
4. SentenceTransformer embeddings for improved semantic understanding
5. UMAP visualization of embedding spaces
6. Enhanced retrieval through augmented queries

## Features

### Basic RAG Features
- **Document Processing**: Automatically loads and processes `.txt` files from the data directory
- **Text Chunking**: Splits large documents into smaller chunks with configurable overlap
- **Vector Embeddings**: Uses OpenAI's `text-embedding-3-small` model for semantic search
- **Persistent Storage**: ChromaDB provides persistent vector storage
- **Contextual Responses**: Generates responses in the style of ancient philosophers (Socrates, Plato, Aristotle)

### Query Expansion RAG Features
- **PDF Processing**: Extracts and processes text from PDF documents using PyPDF
- **Advanced Text Splitting**: Uses LangChain's RecursiveCharacterTextSplitter and SentenceTransformersTokenTextSplitter
- **Query Expansion**: Generates hypothetical answers to improve retrieval accuracy
- **SentenceTransformer Embeddings**: Uses sentence-transformers for better semantic understanding
- **Embedding Visualization**: UMAP projections for visualizing embedding spaces
- **Enhanced Retrieval**: Combines original queries with generated hypothetical answers

## Project Structure

```
.
├── .env                       # Environment variables (OpenAI API key)
├── basic-rag.py              # Basic RAG implementation
├── query-expansion-rag.py    # Advanced RAG with query expansion
├── helper_utils.py           # Utility functions for projections and formatting
├── requirements.txt          # Python dependencies
├── data/                     # Documents directory
│   ├── discourses.txt        # Epictetus discourses
│   ├── meditations.txt       # Marcus Aurelius meditations
│   └── progit.pdf            # Pro Git book PDF for query expansion demo
└── chroma_storage/           # ChromaDB persistent storage
    └── ...                  # Database files
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
   Place your `.txt` files and PDF documents in the `data/` directory

## Usage

### Basic RAG

Run the basic RAG system:

```bash
python basic-rag.py
```

The script will:
1. Load documents from the `data/` directory
2. Process and chunk the text
3. Generate embeddings and store them in ChromaDB
4. Execute an example query about Stoic philosophy
5. Return a philosophically-styled response

### Query Expansion RAG

Run the advanced RAG system with query expansion:

```bash
python query-expansion-rag.py
```

This script demonstrates:
1. PDF document processing and text extraction
2. Advanced text chunking strategies
3. Query expansion through hypothetical answer generation
4. Enhanced retrieval using augmented queries
5. Visualization of embeddings in 2D space using UMAP
6. Technical document analysis with improved accuracy

### Key Functions

#### Basic RAG ([`basic-rag.py`](basic-rag.py))
- **[`load_documents_from_directory()`](basic-rag.py)**: Loads all `.txt` files from a directory
- **[`split_text()`](basic-rag.py)**: Splits text into chunks with configurable size and overlap
- **[`get_openai_embedding()`](basic-rag.py)**: Generates embeddings using OpenAI API
- **[`query_documents()`](basic-rag.py)**: Retrieves relevant document chunks for a query
- **[`generate_response()`](basic-rag.py)**: Generates contextual responses using retrieved chunks

#### Query Expansion RAG ([`query-expansion-rag.py`](query-expansion-rag.py))
- **PDF Processing**: Uses PyPDF to extract text from PDF documents
- **[`augment_query_generated()`](query-expansion-rag.py)**: Generates hypothetical answers to expand queries
- **Advanced Text Splitting**: LangChain's RecursiveCharacterTextSplitter and token-based splitting
- **SentenceTransformer Embeddings**: Uses sentence-transformers for semantic understanding
- **UMAP Visualization**: Projects high-dimensional embeddings to 2D for visualization
- **Enhanced Retrieval**: Combines original and augmented queries for better results

## Configuration

### Basic RAG Settings
You can modify these settings in [basic-rag.py](basic-rag.py):

- `chunk_size`: Size of text chunks (default: 1000 characters)
- `chunk_overlap`: Overlap between chunks (default: 20 characters)
- `n_results`: Number of relevant chunks to retrieve (default: 2)

### Query Expansion RAG Settings
You can modify these settings in [query-expansion-rag.py](query-expansion-rag.py):

- `chunk_size`: Character-level chunk size (default: 1000)
- `tokens_per_chunk`: Token-level chunk size (default: 256)
- `n_results`: Number of relevant chunks to retrieve (default: 5)
- UMAP parameters for embedding visualization

## Example Queries

### Basic RAG Example
The default example asks: *"What does Epictetus say our chief task in life is as Stoics"*

The system will:
1. Find relevant chunks from the stored documents
2. Generate a response in the style of ancient philosophers
3. Provide concise, wisdom-filled answers with rhetorical questions and analogies

### Query Expansion RAG Example
The advanced system demonstrates technical document analysis with queries about Git and software development:
- *"What are the main Git workflow strategies?"*
- *"How do you handle merge conflicts in Git?"*
- *"What are the best practices for branching?"*

The system will:
1. Generate hypothetical answers to expand the query context
2. Retrieve more relevant document sections using the augmented query
3. Visualize the embedding space to show query and document relationships
4. Provide more accurate technical information from the Pro Git book

## Dependencies

### Core Dependencies
- `chromadb`: Vector database for storing and querying embeddings
- `openai`: OpenAI API client for embeddings and chat completions
- `python-dotenv`: Environment variable management

### Query Expansion RAG Additional Dependencies
- `pypdf`: PDF document processing and text extraction
- `langchain`: Advanced text splitting and processing utilities
- `langchain-text-splitters`: Specialized text splitting components
- `sentence-transformers`: Local embedding generation
- `umap-learn`: Dimensionality reduction for embedding visualization
- `matplotlib`: Plotting and visualization of embedding spaces
- `numpy`: Numerical computing support

## Learning Notes

### Basic RAG Implementation
This demonstrates:
- Basic RAG architecture and workflow
- Vector similarity search with embeddings
- Document chunking strategies
- Integration between different AI services
- Persistent vector storage

### Query Expansion RAG Implementation
This advanced approach demonstrates:
- Hypothetical Document Embeddings (HyDE) technique
- Query expansion strategies for improved retrieval
- Advanced text processing with LangChain
- Local vs. API-based embedding approaches
- Embedding space visualization techniques
- Multi-modal document processing (text and PDF)
