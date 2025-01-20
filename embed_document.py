from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Document
from datasets import load_dataset
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
import pickle
import os
import json
import random 


def embed_documents(embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    dataset = load_dataset("./data", split="test")
    # TODO: optimize this and deduplicate
    docs = [Document(content=doc["text_description"],
                     meta={"filename": doc["image_filename"]}) for doc in dataset]

    document_store = InMemoryDocumentStore()

    doc_embedder = SentenceTransformersDocumentEmbedder(model=embedding_model)
    doc_embedder.warm_up()

    docs_with_embeddings = doc_embedder.run(docs)
    document_store.write_documents(
        docs_with_embeddings["documents"], policy=DuplicatePolicy.SKIP)
    return document_store


def save_database_to_disk(database, path: str, name: str) -> None:
    """
    Function to validate the path and call the save_to_disk method.

    :param database: The database object with a `save_to_disk` method.
    :param path: The path to save the database, including the file name.
    """
    # Validate the provided path
    if os.path.isdir(path):
        path = os.path.join(path, name)

    # Ensure the path ends with a .json file extension (btw it wont cause any problems if it doesnt)
    if not path.endswith(".json"):
        raise ValueError("The file path must end with '.json'.")

    # Call the method
    database.save_to_disk(path)


# ds = embed_documents()
# save_database_to_disk(ds, path='./data')

def embed_documents_grouped(embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    with open("./data/texts.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    docs = [
        Document(content=doc)
        for doc in dataset
    ]
    document_store = InMemoryDocumentStore()
    doc_embedder = SentenceTransformersDocumentEmbedder(model=embedding_model)
    doc_embedder.warm_up()
    docs_with_embeddings = doc_embedder.run(docs)
    document_store.write_documents(
        docs_with_embeddings["documents"], policy=DuplicatePolicy.SKIP)
    return document_store


filtered_docs = embed_documents_grouped()
save_database_to_disk(filtered_docs, path='./data', name='DocMerged.json')


def load_json_file(file_path: str) -> dict:
    """Reads a JSON file and returns its contents as a Python dictionary."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON file.")
        return {}


def extract_document_contents(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Extract the content from each document
    document_contents = [Document(content=doc["content"])
                         for doc in data.get("documents", [])]

    return document_contents


def load_document_store_with_embeddings(file_path: str) -> InMemoryDocumentStore:
    """
    Load embeddings and documents from a JSON file into an InMemoryDocumentStore.

    :param file_path: Path to the JSON file containing the document store data.
    :return: An initialized InMemoryDocumentStore with documents having embeddings.
    """
    document_store = InMemoryDocumentStore()

    with open(file_path, "r") as file:
        data = json.load(file)

    documents = data.get("documents", [])
    print(f"Number of documents in the JSON: {len(documents)}")

    documents_with_embeddings = [
        Document(
            content=doc.get("content"),
            embedding=doc.get("embedding"),
            meta=doc.get("meta", {})
        )
        for doc in documents if "embedding" in doc
    ]
    print(f"Number of documents with embeddings: {
          len(documents_with_embeddings)}")

    # Write documents into the document store
    document_store.write_documents(documents_with_embeddings)

    return document_store

def split_list_data(data_list, val_ratio=0.8, test_ratio=0.2, seed=42):
    # Shuffle the data to ensure randomness
    random.seed(seed)
    random.shuffle(data_list)

    val_size = int(len(data_list) * val_ratio)
    test_size = len(data_list) - val_size

    val_data = data_list[:val_size]  # First 80% for validation
    test_data = data_list[val_size:]  # Last 20% for testing

    return val_data, test_data

