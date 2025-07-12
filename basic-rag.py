import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
from typing import cast
from chromadb.api.types import EmbeddingFunction

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-small"
)

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, 
    embedding_function=cast(EmbeddingFunction, openai_ef)
)

client = OpenAI()

def load_documents_from_directory(directory_path):
    print("==== Loading documents from data directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

#Chucking settings
chunk_size=1000
chunk_overlap=20

# Function to split text into chunks
def split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Load documents from the directory
directory_path = "./data"
documents = load_documents_from_directory(directory_path)

# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

# Function to generate embeddings using OpenAI API
def get_openai_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    print("==== Generating embeddings... ====")
    return embedding

# Generate embeddings for the document chunks
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_openai_embedding(doc["text"])

# Upsert documents with embeddings into Chroma
for doc in chunked_documents:
    print("==== Inserting chunks into db;;; ====")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
    )

# Function to query documents
def query_documents(question, n_results=2):
    # query_embedding = get_openai_embedding(question)
    results = collection.query(query_texts=[question], n_results=n_results)

    documents = results.get("documents", [])
    if documents:
        relevant_chunks = [doc for sublist in documents for doc in sublist]
    else:
        relevant_chunks = []
    
    print("==== Returning relevant chunks ====")
    return relevant_chunks
 
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are a wise ancient philosopher, steeped in the traditions of Socrates, Plato, and Aristotle. "
        "When answering questions, speak with the thoughtful deliberation and profound wisdom of antiquity. "
        "Use the following pieces of retrieved context to contemplate and answer the question with philosophical insight. "
        "If the knowledge escapes you, acknowledge this limitation with humility, as Socrates would say 'I know that I know nothing.' "
        "Frame your response with rhetorical questions, analogies to nature or the human condition, and timeless wisdom. "
        "Keep your answer concise yet profound, using no more than three sentences."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message
    return answer

# Example query
question = "What does Epictetus say our chief task in life is as Stoics"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)



