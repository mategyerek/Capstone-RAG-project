from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Document
from datasets import load_dataset
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
import pickle
import os 
import json
import json
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from getpass import getpass
import os
import pickle
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Document
from datasets import load_dataset
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy


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
    print(f"Number of documents with embeddings: {len(documents_with_embeddings)}")

    # Write documents into the document store
    document_store.write_documents(documents_with_embeddings)

    return document_store


embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
text_embedder = SentenceTransformersTextEmbedder(
    model=embedding_model)

retriever = InMemoryEmbeddingRetriever(load_document_store_with_embeddings(file_path= '/Users/stefanbozhilov/Documents/GitHub/Capstone-RAG-project/data/DocStore.json'))

template = [ChatMessage.from_user("""
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
""")]

prompt_builder = ChatPromptBuilder(template=template)

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "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"
chat_generator = OpenAIChatGenerator(model="gpt-4o-mini")

basic_rag_pipeline = Pipeline()
# Add components to your pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", chat_generator)

# Now, connect the components to each other
basic_rag_pipeline.connect("text_embedder.embedding",
                           "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder")
basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

print(basic_rag_pipeline)
question = "Which car model from Aston Martin is categorized as a Subcompact Car?"

response = basic_rag_pipeline.run(
    {"text_embedder": {"text": question}, "prompt_builder": {"question": question}}, include_outputs_from= ['text_embedder', 'retriever', 'prompt_builder', 'llm'
    ])

print(response["llm"]["replies"][0].text)
