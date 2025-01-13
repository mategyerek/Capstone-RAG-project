from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
import pickle
import os
import json
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder, AnswerBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder


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


if __name__ == "__main__":
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    text_embedder = SentenceTransformersTextEmbedder(
        model=embedding_model)

    retriever = InMemoryEmbeddingRetriever(
        load_document_store_with_embeddings(file_path='./data/DocStore.json'))

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
    basic_rag_pipeline.add_component("generator", chat_generator)
    basic_rag_pipeline.add_component("answer_builder", AnswerBuilder())

    # Now, connect the components to each other
    basic_rag_pipeline.connect("text_embedder.embedding",
                               "retriever.query_embedding")
    basic_rag_pipeline.connect("retriever", "prompt_builder")
    basic_rag_pipeline.connect("prompt_builder.prompt", "generator.messages")
    basic_rag_pipeline.connect("generator.replies", "answer_builder.replies")
    basic_rag_pipeline.connect("retriever", "answer_builder.documents")

    print(basic_rag_pipeline)
    questions = ["Which car model from Aston Martin is categorized as a Subcompact Car?",
                 "Which car model from Ferrari is categorized as a Subcompact Car?"]
    responses = []
    answers = []
    documents = []
    for question in questions:
        response = basic_rag_pipeline.run(
            {"text_embedder": {"text": question}, "prompt_builder": {"question": question}, "answer_builder": {"query": question}}, include_outputs_from={"retriever"})
        responses.append(response)
        print(response["retriever"])
        answers.append(response["answer_builder"]["answers"][0].data)
        current_docs = response["answer_builder"]["answers"][0].documents
        documents.append([doc.content for doc in current_docs])

print(len(questions), len(answers), len(documents))

with open("./data/debug.pickle", "wb") as f:
    pickle.dump([questions, answers, documents], f, pickle.HIGHEST_PROTOCOL)
