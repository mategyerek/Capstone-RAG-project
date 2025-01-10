from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Document
from datasets import load_dataset
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
import pickle
import os
import json


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


def save_database_to_disk(database, path: str) -> None:
    """
    Function to validate the path and call the save_to_disk method.

    :param database: The database object with a `save_to_disk` method.
    :param path: The path to save the database, including the file name.
    """
    # Validate the provided path
    if os.path.isdir(path):
        path = os.path.join(path, "DocStore.json")

    # Ensure the path ends with a .json file extension
    if not path.endswith(".json"):
        raise ValueError("The file path must end with '.json'.")

    # Call the method
    database.save_to_disk(path)


#ds = embed_documents()
##ave_database_to_disk(ds, path='./data')

def embed_documents_filtered(embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    with open("./data/texts.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    docs = [
        Document(content=" ".join(doc_group))  # Combine all text segments into one string
        for doc_group in dataset
    ]
    document_store = InMemoryDocumentStore()
    doc_embedder = SentenceTransformersDocumentEmbedder(model=embedding_model)
    doc_embedder.warm_up()
    docs_with_embeddings = doc_embedder.run(docs)
    document_store.write_documents(
        docs_with_embeddings["documents"], policy=DuplicatePolicy.SKIP)
    return document_store


filtered_docs = embed_documents_filtered()
save_database_to_disk(filtered_docs, path='./data')



