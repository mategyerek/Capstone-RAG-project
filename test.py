from haystack.document_stores.in_memory import InMemoryDocumentStore

from datasets import load_dataset
from haystack import Document


document_store = InMemoryDocumentStore()

dataset = load_dataset("./data", split="test")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
print(docs[0])
