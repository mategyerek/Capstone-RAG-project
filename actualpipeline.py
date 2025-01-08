from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
import os
import pickle
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy


embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
document_store = InMemoryDocumentStore()

with open("data/embedded_docs.pickle", 'rb') as f:
    docs = pickle.load(f)

doc_embedder = SentenceTransformersDocumentEmbedder(model=embedding_model)
doc_embedder.warm_up()

docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(
    docs_with_embeddings["documents"], policy=DuplicatePolicy.SKIP)


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
question = "What types of road improvements are planned for Greenmount Road in Belleville, Illinois?"

response = basic_rag_pipeline.run(
    {"text_embedder": {"text": question}, "prompt_builder": {"question": question}})

print(response["llm"]["replies"][0].text)

print(response["retriever"])
