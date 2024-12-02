import os
import tempfile
import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from github import Github, Repository
from git import Repo
from openai import OpenAI
from pathlib import Path
from langchain.schema import Document
from pinecone import Pinecone
from dotenv import load_dotenv
import ast
import re

load_dotenv()

def get_pinecone_index():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is missing from the environment variables.")

    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index("codebase-rag")
    
    return pinecone_index

pinecone_index = get_pinecone_index()


groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing from the environment variables.")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key
)

app = FastAPI()

origins = [
    "http://localhost:5173" # change this to the server later
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get_root(pinecone_index: Pinecone = Depends(get_pinecone_index)):
    # access the pinecone_index
    return {"message": "Pinecone initialized successfully!"}

# clone the repo
def clone_repository(repo_url):
    """Clones a GitHub repository to a temporary directory.

    Args:
        repo_url: The URL of the GitHub repository.

    Returns:
        The path to the cloned repository.
    """
    repo_name = repo_url.split("/")[-1]  # Extract repository name from URL
    repo_path = f"/content/{repo_name}"
    Repo.clone_from(repo_url, str(repo_path))
    return str(repo_path)

# path = clone_repository("https://github.com/CoderAgent/SecureAgent")


# Endpoint to clone a repository
@app.post("/clone-repository/")
async def clone_repo(repo_url: str):
    try:
        repo_path = clone_repository(repo_url)
        return {"message": f"Repository cloned to {repo_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java',
                         '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}

IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
'__pycache__', '.next', '.vscode', 'vendor'}



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
        print(f"Error processing file {file_path}: {str(e)}")
        return None


# Endpoint to get file content from a repository
@app.post("/get-files-content/")
async def get_files_content(repo_url: str):
    try:
        repo_path = clone_repository(repo_url)
        files_content = get_main_files_content(repo_path)
        return {"files": files_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        for root, _, files in os.walk(repo_path):
            # Skip if current directory is in ignored directories
            if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
                continue

            # Process each file in current directory
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(file_path, repo_path)
                    if file_content:
                        files_content.append(file_content)

    except Exception as e:
        print(f"Error reading repository: {str(e)}")

    return files_content


def extract_py_functions(content: str):
    """
    Extract function from python files

    Args: content (str): code of the python file

    Returns: List of function names
    """
    try:
        tree = ast.parse(content)
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    except Exception as e:
        print(f"Error extracting functions: {str(e)}")
        return []


# Endpoint to extract Python functions from a file in a repository
@app.post("/extract-py-functions/")
async def extract_functions_py(repo_url: str, file_name: str):
    try:
        repo_path = clone_repository(repo_url)
        file_content = get_file_content(os.path.join(repo_path, file_name), repo_path)
        if file_content:
            functions = extract_py_functions(file_content["content"])
            return {"functions": functions}
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def extract_ts_functions(content: str):
    """
    Extract function from TypeScript files, including function expressions

    Args: content (str): code of the TypeScript file

    Returns: List of function names
    """
    pattern = r"""
        (?:export\s+default\s+|export\s+|async\s+)?         # keywords 'export', 'export default', 'async', etc.
        function\s+(\w+)\s*\(                               # find the function name before '('
        |                                                   # or
        (\w+)\s*=\s*function\s*(\w*)\s*\(                   # find function expressions 'const bee = function() {...}'
    """
    all_matches = re.findall(pattern, content, flags=re.VERBOSE)

    functions = []
    for match in all_matches:
        if match[0]:
            functions.append(match[0])
        elif match[1]:
            functions.append(match[1])
        elif match[2]:
            functions.append(match[2])

    return functions

# Endpoint to extract TypeScript functions from a file in a repository
@app.post("/extract-ts-functions/")
async def extract_functions_ts(repo_url: str, file_name: str):
    try:
        repo_path = clone_repository(repo_url)
        file_content = get_file_content(os.path.join(repo_path, file_name), repo_path)
        if file_content:
            functions = extract_ts_functions(file_content["content"])
            return {"functions": functions}
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

@app.post("/process-repository/")
async def process_repository(repo_url: str):
    try:
        # Clone the repository each time it's requested
        repo_path = clone_repository(repo_url)

        # Extract file content
        file_content = get_main_files_content(repo_path)

        # Get HuggingFace embeddings
        documents = []

        for file in file_content:
            doc = Document(
                page_content=f"{file['name']}\n{file['content']}",
                metadata={"source": file['name']}
            )
            documents.append(doc)

        # Extract functions
        function_documents = []

        for file in file_content:
            file_name = file["name"]
            file_content = file["content"]
            
            # Extract Python functions
            if file_name.endswith('.py'):
                functions = extract_py_functions(file_content)
                
                for func in functions:
                    doc = Document(
                        page_content=f"Function: {func}\n\nContent:\n{file_content}",
                        metadata={"source": file_name, "function": func}
                    )
                    function_documents.append(doc)
            
            # Extract TypeScript functions
            elif file_name.endswith('.ts') or file_name.endswith('.tsx'):
                functions = extract_ts_functions(file_content)
                
                for func in functions:
                    doc = Document(
                        page_content=f"Function: {func}\n\nContent:\n{file_content}",
                        metadata={"source": file_name, "function": func}
                    )
                    function_documents.append(doc)

        # Combine documents and function documents
        all_documents = documents + function_documents

        # Initialize Pinecone vector store
        vectorstore = PineconeVectorStore.from_documents(
            documents=all_documents,
            embedding=HuggingFaceEmbeddings(),
            index_name="codebase-rag",
            namespace="https://github.com/CoderAgent/SecureAgent"
        )

        # Return a success response
        return {"message": "Repository processed successfully!"}
    except Exception as e:
        # Handle any errors during the process
        raise HTTPException(status_code=500, detail=str(e))


def perform_rag(query):
    raw_query_embedding = get_huggingface_embeddings(query)

    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=5, include_metadata=True, namespace="https://github.com/CoderAgent/SecureAgent")

    # Get the list of retrieved texts
    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    # Modify the prompt below as need to improve the response quality
    system_prompt = f"""
        You are a Senior Software Engineer, specializing in TypeScript. 
        Answer any questions I have about the codebase, based on the code provided. 
        Always consider all of the context provided when forming a response.
        Let's think step by step. Verify step by step.
    """
    try:
        llm_response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ]
        )
        response = llm_response.choices[0].message.content
    except Exception as e:
        print(f"Error occurred: {e}")
        # Retry or handle the error with a larger token space llm
        llm_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ]
        )
        response = llm_response.choices[0].message.content
    
    return response

@app.post("/query")
async def query_codebase(query: str):
    try:
        # Perform the RAG query
        response = perform_rag(query)
        return {"message": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
