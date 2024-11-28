# main.py

import os
import tempfile
import shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from git import Repo
import pinecone
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Pinecone as PineconeVectorStore  # Updated import
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import openai
from dotenv import load_dotenv

import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if present
load_dotenv()

app = FastAPI()

# Define request models
class QueryRequest(BaseModel):
    query: str

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "codebase-rag"

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    logger.error("Pinecone API key or environment not set in environment variables.")
    raise ValueError("Pinecone API key or environment not set.")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Check if index exists, if not, create it
if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    # Assuming the embedding dimension is 768 for 'sentence-transformers/all-mpnet-base-v2'
    pinecone.create_index(name=PINECONE_INDEX_NAME, dimension=768, metric="cosine")

pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)

# Initialize Embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize OpenAI
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ API key not set in environment variables.")
    raise ValueError("GROQ API key not set.")
openai.api_key = GROQ_API_KEY

# Supported extensions and ignored directories
SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java',
                        '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}

IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
               '__pycache__', '.next', '.vscode', 'vendor'}

# Define the GitHub repository to clone
REPO_URL = "https://github.com/CoderAgent/SecureAgent"

def clone_repository(repo_url):
    """Clones a GitHub repository to a temporary directory.

    Args:
        repo_url: The URL of the GitHub repository.

    Returns:
        The path to the cloned repository.
    """
    repo_name = repo_url.rstrip('/').split("/")[-1]  # Extract repository name from URL
    temp_dir = tempfile.mkdtemp()
    repo_path = os.path.join(temp_dir, repo_name)
    logger.info(f"Cloning repository {repo_url} into {repo_path}")
    Repo.clone_from(repo_url, repo_path)
    return repo_path

def get_file_content(file_path, repo_path):
    """
    Get content of a single file.

    Args:
        file_path (str): Path to the file

    Returns:
        Optional[Dict[str, str]]: Dictionary with file name and content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Get relative path from repo root
        rel_path = os.path.relpath(file_path, repo_path)

        return {
            "name": rel_path,
            "content": content
        }
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return None

def get_main_files_content(repo_path: str):
    """
    Get content of supported code files from the local repository.

    Args:
        repo_path: Path to the local repository

    Returns:
        List of dictionaries containing file names and contents
    """
    files_content = []

    try:
        for root, dirs, files in os.walk(repo_path):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(file_path, repo_path)
                    if file_content:
                        files_content.append(file_content)

    except Exception as e:
        logger.error(f"Error reading repository: {str(e)}")

    return files_content

def get_huggingface_embeddings(text):
    """Generate embeddings for the given text using HuggingFace model."""
    return embedding_model.encode(text).tolist()

def perform_rag(query: str) -> str:
    """Performs Retrieval-Augmented Generation based on the query."""
    raw_query_embedding = get_huggingface_embeddings(query)

    top_matches = pinecone_index.query(
        vector=raw_query_embedding,
        top_k=5,
        include_metadata=True,
        namespace="https://github.com/CoderAgent/SecureAgent"
    )

    # Extract contexts
    contexts = [match['metadata'].get('content', '') for match in top_matches['matches']]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    system_prompt = """You are a Senior Software Engineer, specializing in TypeScript.

Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Replace with your desired model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    return response.choices[0].message['content'].strip()

# Initial setup: Clone repo, process files, and populate Pinecone
@app.on_event("startup")
def startup_event():
    logger.info("Starting up and setting up Pinecone index.")
    repo_path = clone_repository(REPO_URL)
    file_content = get_main_files_content(repo_path)

    documents = []
    for file in file_content:
        doc = Document(
            page_content=f"{file['name']}\n{file['content']}",
            metadata={"source": file['name'], "content": file['content']}
        )
        documents.append(doc)

    # Generate embeddings and prepare for upsert
    vectors = []
    for doc in documents:
        embedding = hf_embeddings.embed_query(doc.page_content)
        vectors.append((doc.metadata['source'], embedding, doc.metadata))

    # Upsert vectors to Pinecone
    logger.info(f"Upserting {len(vectors)} vectors to Pinecone.")
    pinecone_index.upsert(vectors=vectors, namespace="https://github.com/CoderAgent/SecureAgent")

    # Cleanup cloned repository to save space
    shutil.rmtree(os.path.dirname(repo_path))
    logger.info("Startup setup completed.")

# Define the RAG endpoint
@app.post("/perform_rag")
async def perform_rag_endpoint(request: QueryRequest):
    try:
        response = perform_rag(request.query)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in perform_rag_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))