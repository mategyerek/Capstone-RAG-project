from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Document
from datasets import load_dataset
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
import pickle
import os


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


ds = embed_documents()
save_database_to_disk(ds, path='./data')
