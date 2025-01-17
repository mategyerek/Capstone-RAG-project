from haystack.evaluation.eval_run_result import EvaluationRunResult
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
import os
import json
from haystack import Pipeline
from haystack.components.generators import HuggingFaceAPIGenerator
import os
from haystack.dataclasses import ChatMessage
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.evaluators.document_mrr import DocumentMRREvaluator
from haystack.components.evaluators.faithfulness import FaithfulnessEvaluator
from haystack.components.evaluators.sas_evaluator import SASEvaluator
from embed_document import load_json_file, extract_document_contents
from haystack.utils import ComponentDevice
import pandas as pd 
from bert_score import score
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

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
        load_document_store_with_embeddings(file_path='./data/DocMerged.json'))

    template = """
    Given a context, provide ONLY the answers to the questions without repepating the question.
    Keep the answers very short, only limiting yourself to directly answering the question. 
    DO NOT GENERATE MORE QUESTIONS AND DO NOT GENERATE MORE ANSWERS. 
    
    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:                                  
    """

    prompt_builder = PromptBuilder(template=template)

    if "OPENAI_API_KEY" not in os.environ:
         os.environ["OPENAI_API_KEY"] = "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"
    if "HF_API_TOKEN" not in os.environ:
        os.environ["HF_API_TOKEN"] = "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"
    # BEWARE THIS WILL BREAK!
    generator = HuggingFaceAPIGenerator(api_type="serverless_inference_api",
                                        api_params={"model": "mistralai/Mistral-7B-Instruct-v0.3"},
                                        generation_kwargs={'temperature': 1.25})

    basic_rag_pipeline = Pipeline()
    # Add components to your pipeline
    basic_rag_pipeline.add_component("text_embedder", text_embedder)
    basic_rag_pipeline.add_component("retriever", retriever)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("generator", generator)
    basic_rag_pipeline.add_component("answer_builder", AnswerBuilder())

    # Now, connect the components to each other
    basic_rag_pipeline.connect("text_embedder.embedding",
                               "retriever.query_embedding")
    basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
    basic_rag_pipeline.connect("prompt_builder", "generator")
    basic_rag_pipeline.connect("generator.replies", "answer_builder.replies")
    basic_rag_pipeline.connect("retriever", "answer_builder.documents")

    print(basic_rag_pipeline)

    # Setting up Evaluation Pipeline
    questions = load_json_file('data/querys.json')
    ground_truth_answers = load_json_file('data/answers.json')
    all_documents = extract_document_contents('./data/DocMerged.json')


    with open("./data/doc_lookup.json", "r") as f:
        lookup_table = json.load(f)

    ground_truth_docs = [all_documents[lookup_table.get(
        str(i))] for i in range(len(questions))]

    print(len(questions), len(ground_truth_answers),
        len(ground_truth_docs))  # 100 100 100 respectively

    eval_pipeline = Pipeline()
    eval_pipeline.add_component("doc_mrr_evaluator", DocumentMRREvaluator())
    eval_pipeline.add_component("faithfulness", FaithfulnessEvaluator())
    eval_pipeline.add_component("sas_evaluator", SASEvaluator(
        model="sentence-transformers/all-MiniLM-L6-v2"))


    rag_answers = []
    retrieved_docs = []

    for question in list(questions):
        response = basic_rag_pipeline.run(
            {
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question},
            }
        )
        # print(f"Documents: {response["answer_builder"]["documents"]}")
        print(f"Question: {question}")
        print("Answer from pipeline:")
        print(response["answer_builder"]["answers"][0].data)
        print("\n-----------------------------------\n")

        rag_answers.append(response["answer_builder"]["answers"][0].data)
        retrieved_docs.append(response["answer_builder"]["answers"][0].documents)


    results = eval_pipeline.run(
        {
            "doc_mrr_evaluator": {
                "ground_truth_documents": list([d] for d in ground_truth_docs),
                "retrieved_documents": retrieved_docs,
            },
            "faithfulness": {
                "questions": list(questions),
                "contexts": list([d] for d in ground_truth_docs),
                "predicted_answers": rag_answers,
            },
            "sas_evaluator": {"predicted_answers": rag_answers, "ground_truth_answers": list(ground_truth_answers)},
        }
    )


    inputs = {
        "question": list(questions),
        "contexts": list([d] for d in ground_truth_docs),
        "answer": list(ground_truth_answers),
        "predicted_answer": rag_answers,
    }



    evaluation_result = EvaluationRunResult(
        run_name="pubmed_rag_pipeline", inputs=inputs, results=results)
    evaluation_result.score_report()
    results_df = evaluation_result.to_pandas()

    #BERT Score Implementation
    results_df['answer'] = results_df['answer']

    def compute_bertscore(row):
        prediction = row['predicted_answer']
        reference = row['answer']
        
        # Compute BERTScore for prediction and reference
        results = score([prediction], [reference], lang='en')

        # Extract precision, recall, and f1 from the results
        precision = results[0][0]  
        recall = results[1][0]     
        f1 = results[2][0]         
        
        return pd.Series([precision, recall, f1])

    # Apply the function to the DataFrame row-wise with axis=1
    results_df[['precision', 'recall', 'f1']] = results_df.apply(compute_bertscore, axis=1)

    results_df.to_csv('./results/results_all-MiniLM-L6-v2_gemma-2-2b-it.csv', index=False)
    print('Evaluation results saved!')



