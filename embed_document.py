from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Document
from datasets import load_dataset
from haystack.document_stores.in_memory import InMemoryDocumentStore
import pickle


dataset = load_dataset("./data", split="test")
# TODO: optimize this and deduplicate
docs = [Document(content=doc["text_description"],
                 meta={"filename": doc["image_filename"]}) for doc in dataset]

with open("data/embedded_docs.pickle", 'wb') as file:
    pickle.dump(docs, file, pickle.HIGHEST_PROTOCOL)
