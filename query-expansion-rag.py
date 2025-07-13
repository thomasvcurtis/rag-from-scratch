from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv

from pypdf import PdfReader
import umap

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

#Note investigate PDF, I was not able to extract the data the text from the design patterns pdf 
reader = PdfReader("data/progit.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

pdf_texts = [text for text in pdf_texts if text]

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)


embedding_function = SentenceTransformerEmbeddingFunction()

chroma_client = chromadb.PersistentClient(path="chroma_storage")
try:
    chroma_collection = chroma_client.get_collection("rag-collection")
except:
    chroma_collection = chroma_client.create_collection("rag-collection")

# extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_texts))]
embeddings = embedding_function(token_split_texts)
chroma_collection.add(ids=ids, documents=token_split_texts, embeddings=embeddings)


query = "How do git merge conflicts work and what are the best strategies for resolving them in collaborative development?"

query_embedding = embedding_function([query])
results = chroma_collection.query(
    query_embeddings=query_embedding, n_results=5, include=["documents"]
)

if results and results["documents"]:
    retrieved_documents = results["documents"][0]
else:
    retrieved_documents = []


def augment_query_generated(query, model="gpt-4.1-nano-2025-04-14"):
    prompt = "You are a helpful expert software development"
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

original_query = "What are the differences between git rebase and git merge, and when should each be used in a team development workflow?"
hypothetical_answer = augment_query_generated(original_query)

joint_query = f"{original_query} {hypothetical_answer}"

joint_query_embedding = embedding_function([joint_query])
results = chroma_collection.query(
    query_embeddings=joint_query_embedding, n_results=5, include=["documents", "embeddings"]
)

if results and results["documents"]:
    retrieved_documents = results["documents"][0]
else:
    retrieved_documents = []

embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

if results and results["embeddings"]:
    retrieved_embeddings = results["embeddings"][0]
else:
    retrieved_embeddings = []
original_query_embedding = embedding_function([original_query])
augmented_query_embedding = embedding_function([joint_query])

projected_original_query_embedding = project_embeddings(
    original_query_embedding, umap_transform
)
projected_augmented_query_embedding = project_embeddings(
    augmented_query_embedding, umap_transform
)
projected_retrieved_embeddings = project_embeddings(
    retrieved_embeddings, umap_transform
)

import matplotlib.pyplot as plt

# Plot the projected query and retrieved documents in the embedding space
plt.figure()

plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
)
plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # display the plot


