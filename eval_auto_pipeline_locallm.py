from haystack.evaluation.eval_run_result import EvaluationRunResult
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
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
from embed_document import load_json_file, extract_document_contents, embed_documents_grouped, save_database_to_disk, load_document_store_with_embeddings, split_list_data
from custom_component import QuestionCutter
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

def check_if_results_exist(embedding_model: str, generator_model: str, temperature, repeat_penalty) -> bool:
    file_name = f"results_{embedding_model.replace('/', '_')}_{generator_model.replace('/', '_')}_{temperature}_{repeat_penalty}.csv"
    return os.path.exists(f'./results/{file_name}')
    
def create_pipeline(embedding_model: str, generator_model: str, doc_store_name: str, temperature = 1, prompt: str = None, repeat_penalty = 1.5):
    """
    Create a pipeline dynamically with the given embedding model and generator model.

    :param embedding_model: Embedding model to use.
    :param generator_model: Generator model to use for generating answers.
    :return: The pipeline object.
    """
    text_embedder = SentenceTransformersTextEmbedder(model=embedding_model)

    # Load the document store with embeddings
    document_store = load_document_store_with_embeddings(file_path=f'./data/{doc_store_name}')

    retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k = 1)

    template = prompt

    prompt_builder = PromptBuilder(template=template)

    # Create HuggingFace generator API with the specified model
    if "OPENAI_API_KEY" not in os.environ:
         os.environ["OPENAI_API_KEY"] = "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"
    if "HF_API_TOKEN" not in os.environ:
        os.environ["HF_API_TOKEN"] = "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"
    """chat_generator = HuggingFaceAPIGenerator(api_type="serverless_inference_api",
                                             api_params={"model": generator_model})"""
    chat_generator = LlamaCppGenerator(
                                        model=generator_model,
                                        generation_kwargs={"temperature": temperature, "max_tokens": 128, "repeat_penalty": repeat_penalty},
                                        model_kwargs={"n_gpu_layers": -1})
    question_cutter = QuestionCutter()
    # Initialize the pipeline
    rag_pipeline = Pipeline()

    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("generator", chat_generator)
    rag_pipeline.add_component("answer_builder", AnswerBuilder())
    rag_pipeline.add_component("question_cutter", question_cutter)

    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder")
    rag_pipeline.connect("prompt_builder", "generator")
    rag_pipeline.connect("generator.replies", "question_cutter.in_text")
    rag_pipeline.connect("question_cutter.out_text", "answer_builder.replies")
    rag_pipeline.connect("retriever", "answer_builder.documents")

    return rag_pipeline

def evaluate_pipeline(rag_pipeline, questions, ground_truth_answers, ground_truth_docs):
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
    rag_pipeline.graph._node["generator"]["instance"].model.close()
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

def save_evaluation_results(results_df, embedding_model, generator_model, temperature, repeat_penalty):
    """
    Save the evaluation results as a CSV file.

    :param results_df: The evaluation results DataFrame.
    :param embedding_model: The name of the embedding model.
    :param generator_model: The name of the generator model.
    """
    filename = f"./results/results_{embedding_model.replace("/", "_")}_{generator_model.replace("/", "_")}_{temperature}_{repeat_penalty}.csv"
    results_df.to_csv(filename, index=False)
    print(f'Evaluation results saved as {filename}')

def save_test_data(test_questions, test_answers, embedding_model, generator_model, temperature, repeat_penalty):
    """
    Save the test questions and answers as a JSON file.

    :param test_questions: The list of test questions.
    :param test_answers: The list of test answers.
    :param embedding_model: The name of the embedding model.
    :param generator_model: The name of the generator model.
    :param temperature: The temperature setting for the model.
    :param repeat_penalty: The repeat penalty setting for the model.
    """
    filename = f"./test_data/test_data_{embedding_model.replace('/', '_')}_{generator_model.replace('/', '_')}_{temperature}_{repeat_penalty}.json"

    test_data = {
        'test_questions': test_questions, 
        'test_answers': test_answers
    }

    os.makedirs(os.path.dirname(filename), exist_ok= True)
    with open(filename, 'w') as f: 
        json.dump(test_data, f, indent = 4)
    
    print(f'Test data saved as {filename}')
def run_evaluation_for_models(embedding_models: list, generator_models: list, temperature: float, prompt: str, repeat_penalty: float):
    for embedding_model in embedding_models:
        for generator_model in generator_models:
            try:
                if check_if_results_exist(embedding_model, generator_model, temperature, repeat_penalty):
                    print(f"Results already exist for {embedding_model} and {generator_model}. Skipping...")
                    continue

                # Embed documents only if results do not exist
                doc_store_name = f"DocMerged_{embedding_model.replace('/', '_')}.json"
                if os.path.exists(f'./data/{doc_store_name}'):
                    print(f"Document store {doc_store_name} already exists. Skipping embedding step.")
                else: 
                    filtered_docs = embed_documents_grouped(embedding_model= embedding_model)
                    save_database_to_disk(filtered_docs, path='./data', name=doc_store_name)
                
                rag_pipeline = create_pipeline(embedding_model, generator_model, doc_store_name = doc_store_name, temperature = temperature, prompt = prompt, repeat_penalty= repeat_penalty)
                
                questions = load_json_file('data/querys.json')
                ground_truth_answers = load_json_file('data/answers.json')
                all_documents = extract_document_contents(f'./data/{doc_store_name}')
                
                with open("./data/doc_lookup.json", "r") as f:
                    lookup_table = json.load(f)
                
                ground_truth_docs = [all_documents[lookup_table.get(str(i))] for i in range(len(questions))]
                questions, test_questions = split_list_data(questions, val_ratio= 0.8, test_ratio= 0.2)
                ground_truth_answers, test_answers = split_list_data(ground_truth_answers, val_ratio = 0.8, test_ratio = 0.2)
                save_test_data(test_questions, test_answers, embedding_model, generator_model, temperature, repeat_penalty)
                
                results_df = evaluate_pipeline(rag_pipeline, questions, ground_truth_answers, ground_truth_docs)
                
                save_evaluation_results(results_df, embedding_model, generator_model, temperature, repeat_penalty)
                #rag_pipeline.graph._node["generator"]["instance"].model.close()
            except Exception as e:
                # Log any errors
                error_message = f"Error occurred for {embedding_model} and {generator_model}: {str(e)}"
                print(error_message)

                log_error_to_file(error_message)

if __name__ == "__main__":
    # Define your embedding and generator models here
    embedding_models = [
        # default: "sentence-transformers/all-MiniLM-L6-v2",
        "multi-qa-mpnet-base-cos-v1", # (mean, 420MB)
        "all-mpnet-base-v2", # performance (mean, 420MB)
        "multi-qa-MiniLM-L6-cos-v1", # (mean, 80MB, better at sentence embedding)
        "all-MiniLM-L12-v2", # (mean, 120MB, better at semantic search)
        "multi-qa-mpnet-base-dot-v1", # similarity(CLS pooling, 420MB)
        "sentence-transformers/nli-bert-base-max-pooling", 
    ]
    
    generator_models = [
        "model_weights/Phi-3.5-mini-instruct-IQ3_XS.gguf",
        "model_weights/Llama-3.2-3B-Instruct-Q3_K_L.gguf",  
        "model_weights/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "model_weights/Llama-3.2-3B-Instruct-Q6_K.gguf",
    ]
    
    prompt = """Given a context, provide ONLY the answers to the questions without repepating the question. 
    Keep the answers very short, only limiting yourself to directly answering the question.  
    DO NOT:
    - Generate multiple questions and answers.
    - Answer in multiple sentences.
    - Include irrelevant information.

    For example:
    - Question: What is the capital of France?
    Answer: Paris. (Correct)
    DO NOT ANSWER: Paris. What is the population of Paris? 2.1 million. (This is Incorrect)

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer: """

    run_evaluation_for_models(embedding_models, generator_models, temperature= 2, prompt = prompt, repeat_penalty= 2)  