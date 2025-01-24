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
from embed_document import load_json_file, extract_document_contents, embed_documents_grouped, save_database_to_disk
from embed_document import load_document_store_with_embeddings
import pandas as pd 
from bert_score import score
import logging
import transformers
import itertools
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

def log_error_to_file(error_message: str):
    """
    Log error messages to a separate text file.
    """
    error_log_file = './data/errors.txt'
    with open(error_log_file, 'a') as f:
        f.write(f"{error_message}\n")

def check_if_results_exist(embedding_model: str, generator_model: str) -> bool:
    file_name = f"results_{embedding_model.replace('/', '_')}_{generator_model.replace('/', '_')}.csv"
    return os.path.exists(f'./results/{file_name}')
    
def create_pipeline(embedding_model: str, generator_model: str, doc_store_name: str):
    """
    Create a pipeline dynamically with the given embedding model and generator model.

    :param embedding_model: Embedding model to use.
    :param generator_model: Generator model to use for generating answers.
    :return: The pipeline object.
    """
    text_embedder = SentenceTransformersTextEmbedder(model=embedding_model)

    # Load the document store with embeddings
    document_store = load_document_store_with_embeddings(file_path=f'./data/{doc_store_name}', similarity_function= 'dot_product')

    retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k = 1)

    template = """
    Given a context, provide ONLY the answers to the questions without repepating the question
    Do NOT generate any additional questions or answers. 
    Provide only the direct answer in a concise and factual manner.
    Do not enumerate the answers.
    
    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:                                  
    """

    prompt_builder = PromptBuilder(template=template)

    # Create HuggingFace generator API with the specified model
    if "OPENAI_API_KEY" not in os.environ:
         os.environ["OPENAI_API_KEY"] = "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"
    if "HF_API_TOKEN" not in os.environ:
        os.environ["HF_API_TOKEN"] = "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"
    chat_generator = HuggingFaceAPIGenerator(api_type="serverless_inference_api",
                                             api_params={"model": generator_model})

    # Initialize the pipeline
    rag_pipeline = Pipeline()

    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("generator", chat_generator)
    rag_pipeline.add_component("answer_builder", AnswerBuilder())

    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder")
    rag_pipeline.connect("prompt_builder", "generator")
    rag_pipeline.connect("generator.replies", "answer_builder.replies")
    rag_pipeline.connect("retriever", "answer_builder.documents")

    return rag_pipeline

def evaluate_pipeline(rag_pipeline, questions, ground_truth_answers, ground_truth_docs, visualise = True):
    """
    Evaluate the pipeline using several metrics.

    :param rag_pipeline: The pipeline to evaluate.
    :param questions: The list of questions.
    :param ground_truth_answers: The ground truth answers for evaluation.
    :param ground_truth_docs: The ground truth documents used for retrieval.
    :return: Evaluation results in a pandas DataFrame.
    """
    eval_pipeline = Pipeline()
    eval_pipeline.add_component("doc_mrr_evaluator", DocumentMRREvaluator())
    eval_pipeline.add_component("faithfulness", FaithfulnessEvaluator())
    eval_pipeline.add_component("sas_evaluator", SASEvaluator(model="sentence-transformers/all-MiniLM-L6-v2"))
    if visualise: 
        eval_pipeline.draw(path = '/Users/stefanbozhilov/Documents/GitHub/Capstone-RAG-project/data/evalgraph.png')

    rag_answers = []
    retrieved_docs = []
    counter = 1

    for question in questions:
        response = rag_pipeline.run(
            {
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question},
            }
        )
        counter += 1
        print(f"Question: {question}")
        print("Answer from pipeline:")
        print(response["answer_builder"]["answers"][0].data)
        print(f'Sample number {counter}')
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

    evaluation_result = EvaluationRunResult(run_name="pubmed_rag_pipeline", inputs=inputs, results=results)
    evaluation_result.score_report()
    results_df = evaluation_result.to_pandas()

    # Compute BERTScore
    results_df['answer'] = results_df['answer']

    def compute_bertscore(row):
        prediction = row['predicted_answer']
        reference = row['answer']
        results = score([prediction], [reference], lang='en')

        precision = results[0][0]
        recall = results[1][0]
        f1 = results[2][0]
        
        return pd.Series([precision, recall, f1])

    results_df[['precision', 'recall', 'f1']] = results_df.apply(compute_bertscore, axis=1)

    return results_df

def save_evaluation_results(results_df, embedding_model, generator_model):
    """
    Save the evaluation results as a CSV file.

    :param results_df: The evaluation results DataFrame.
    :param embedding_model: The name of the embedding model.
    :param generator_model: The name of the generator model.
    """
    filename = f"./results/results_{embedding_model}_{generator_model}.csv"
    results_df.to_csv(filename, index=False)
    print(f'Evaluation results saved as {filename}')

def run_evaluation_for_models(embedding_models: list, generator_models: list, visualise = True):
    for embedding_model in embedding_models:
        for generator_model in generator_models:
            try:
                if check_if_results_exist(embedding_model, generator_model):
                    print(f"Results already exist for {embedding_model} and {generator_model}. Skipping...")
                    continue

                # Embed documents only if results do not exist
                doc_store_name = f"DocMerged_{embedding_model.replace('/', '_')}.json"
                if os.path.exists(f'./data/{doc_store_name}'):
                    print(f"Document store {doc_store_name} already exists. Skipping embedding step.")
                else: 
                    filtered_docs = embed_documents_grouped(embedding_model= embedding_model)
                    save_database_to_disk(filtered_docs, path='./data', name=doc_store_name)
                
                rag_pipeline = create_pipeline(embedding_model, generator_model, doc_store_name = doc_store_name)
                
                questions = load_json_file('data/querys.json')
                ground_truth_answers = load_json_file('data/answers.json')
                all_documents = extract_document_contents(f'./data/{doc_store_name}')
                
                with open("./data/doc_lookup.json", "r") as f:
                    lookup_table = json.load(f)
                
                ground_truth_docs = [all_documents[lookup_table.get(str(i))] for i in range(len(questions))]
                
                if visualise: 
                    rag_pipeline.draw(path = '/Users/stefanbozhilov/Documents/GitHub/Capstone-RAG-project/data/pipegraph.png')

                results_df = evaluate_pipeline(rag_pipeline, questions, ground_truth_answers, ground_truth_docs, visualise= True)

                save_evaluation_results(results_df, embedding_model, generator_model)

            except Exception as e:
                # Log any errors
                error_message = f"Error occurred for {embedding_model} and {generator_model}: {str(e)}"
                print(error_message)

                log_error_to_file(error_message)

if __name__ == "__main__":
    # Define your embedding and generator models here
    embedding_models = [
        "multi-qa-mpnet-base-dot-v1",  # Example model 1 
        # Add more models as needed
        
    ]
    
    generator_models = [
        "mistralai/Mistral-7B-Instruct-v0.3",   
    ]
    
    run_evaluation_for_models(embedding_models, generator_models)  