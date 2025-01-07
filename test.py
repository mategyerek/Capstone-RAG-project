from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from getpass import getpass
import os
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Document
from datasets import load_dataset
from haystack.document_stores.in_memory import InMemoryDocumentStore


# Shout out to deepset-ai for the nice tutorial


document_store = InMemoryDocumentStore()

dataset = load_dataset("./data", split="test")
# TODO: optimize this and deduplicate
docs = [Document(content=doc[f"text_description"],
                 meta=doc["image_filename"] + str(i)) for i, doc in enumerate(dataset)]

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
doc_embedder = SentenceTransformersDocumentEmbedder(model=embedding_model)
doc_embedder.warm_up()

docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

text_embedder = SentenceTransformersTextEmbedder(
    model=embedding_model)

retriever = InMemoryEmbeddingRetriever(document_store)

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
question = "What does Rhodes Statue look like?"

response = basic_rag_pipeline.run(
    {"text_embedder": {"text": question}, "prompt_builder": {"question": question}})

print(response["llm"]["replies"][0].text)
