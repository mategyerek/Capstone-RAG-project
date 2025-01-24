"""
Run this to evaluate the final model or to perform a hyperparameter search. Adjust parameters at the bottom.
Can only be run after querydata.py
"""

from haystack.evaluation.eval_run_result import EvaluationRunResult
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
import os
import json
from haystack import Pipeline
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
import traceback
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

def log_error_to_file(error_message: str):
    """
    Log error messages to a separate text file.
    :param error_message: The error message
    """
    error_log_file = './data/errors.txt'
    with open(error_log_file, 'a') as f:
        f.write(f"{error_message}\n")

def check_if_results_exist(embedding_model: str, generator_model: str, temperature, repeat_penalty, similarity) -> bool:
    """
    Check if the results belonging to a certain run exist based on run parameters.
    :param: embedding_model: 
    :param: generator_model
    :param: temperature
    :param: repeat_penalty
    :param: similarity
    :return: A bool indicating whether document exists
    """
    file_name = f"results_{embedding_model.replace('/', '_')}_{generator_model.replace('/', '_')}_{temperature}_{repeat_penalty}_{similarity}.csv"
    return os.path.exists(f'./results/{file_name}')
    
def create_pipeline(embedding_model: str, generator_model: str, doc_store_name: str, temperature = 1, prompt: str = None, repeat_penalty = 1.5, cut_question=False, similarity_function="dot_product"):
    """
    Create a pipeline dynamically with the given embedding model and generator model.

    :param embedding_model: Embedding model to use.
    :param generator_model: Generator model to use for generating answers.
    :return: The pipeline object.
    """
    # Initialize embedder for query embedding
    text_embedder = SentenceTransformersTextEmbedder(model=embedding_model)

    # Load the document store with embeddings
    document_store = load_document_store_with_embeddings(file_path=f'./data/{doc_store_name}', similarity_function=similarity_function)

    # Initialize the retriever
    retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k = 1)

    # set template
    template = prompt

    # Build LLM prompt
    prompt_builder = PromptBuilder(template=template)

    # Define openAI key (only needed for faithfulness evaluation)
    if "OPENAI_API_KEY" not in os.environ:
         os.environ["OPENAI_API_KEY"] = "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"
    # Create HuggingFace generator API with the specified model
    if "HF_API_TOKEN" not in os.environ:
        os.environ["HF_API_TOKEN"] = "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"
    # Uncomment for cloud-based inference
    """chat_generator = HuggingFaceAPIGenerator(api_type="serverless_inference_api",
                                             api_params={"model": generator_model})"""
    # Initialize local LLM comment out for cloud-based inference
    chat_generator = LlamaCppGenerator(
                                        model=generator_model,
                                        generation_kwargs={"temperature": temperature, "max_tokens": 128, "repeat_penalty": repeat_penalty, "stop": ["Question:"] if cut_question else []},
                                        model_kwargs={"n_gpu_layers": -1})
    
    # Initialize the pipeline and add the components to it
    rag_pipeline = Pipeline()

    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("generator", chat_generator)
    rag_pipeline.add_component("answer_builder", AnswerBuilder())
    if cut_question:
        # Optional component to remove hallucinations
        rag_pipeline.add_component("question_cutter", QuestionCutter())

    # Connect pipeline components
    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder")
    rag_pipeline.connect("prompt_builder", "generator")
    rag_pipeline.connect("retriever", "answer_builder.documents")
    if cut_question:
        # Connect optional components
        rag_pipeline.connect("generator.replies", "question_cutter.in_text")
        rag_pipeline.connect("question_cutter.out_text", "answer_builder.replies")
    else:
        # Connect in case optional component is not present
        rag_pipeline.connect("generator.replies", "answer_builder.replies")    

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
    # Initialize pipeline and add components to it
    eval_pipeline = Pipeline()
    eval_pipeline.add_component("doc_mrr_evaluator", DocumentMRREvaluator())
    eval_pipeline.add_component("faithfulness", FaithfulnessEvaluator())
    eval_pipeline.add_component("sas_evaluator", SASEvaluator(model="sentence-transformers/all-MiniLM-L6-v2"))

    # Initialize empty answers and docs array
    rag_answers = []
    retrieved_docs = []
    counter = 1
    # fill up andswers and retrieved docs by running the pipeline
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
    
    # unload the LLM from vram
    rag_pipeline.graph._node["generator"]["instance"].model.close()

    # run the evaluation pipeline
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

    # put data in the right format for creating an evaluation result object
    inputs = {
        "question": list(questions),
        "contexts": list([d] for d in ground_truth_docs),
        "answer": list(ground_truth_answers),
        "predicted_answer": rag_answers,
    }

    # Create and save the evaluation results as a pandas data frame
    evaluation_result = EvaluationRunResult(run_name="pubmed_rag_pipeline", inputs=inputs, results=results)
    evaluation_result.score_report()
    results_df = evaluation_result.to_pandas()

    # Compute BERTScore
    results_df['answer'] = results_df['answer']

    def compute_bertscore(row):
        """
        Compute the BERT score for a row of data

        :param row: pandas dataframe row including 'predicted_answer' and 'answer'
        :return: pandas series with precision, recall and f1 score
        """
        prediction = row['predicted_answer']
        reference = row['answer']
        results = score([prediction], [reference], lang='en')

        precision = results[0][0]
        recall = results[1][0]
        f1 = results[2][0]
        
        return pd.Series([precision, recall, f1])

    # compute BERT score for all rows
    results_df[['precision', 'recall', 'f1']] = results_df.apply(compute_bertscore, axis=1)

    return results_df

def save_evaluation_results(results_df, embedding_model, generator_model, temperature, repeat_penalty, similarity):
    """
    Save the evaluation results as a CSV file.

    :param results_df: The evaluation results DataFrame.
    :param embedding_model: The name of the embedding model.
    :param generator_model: The name of the generator model.
    """
    filename = f"./results/results_{embedding_model.replace("/", "_")}_{generator_model.replace("/", "_")}_{temperature}_{repeat_penalty}_{similarity}.csv"
    results_df.to_csv(filename, index=False)
    print(f'Evaluation results saved as {filename}')

def save_test_data(test_questions, test_answers):
    """
    Save the test questions and answers as a JSON file.

    :param test_questions: The list of test questions.
    :param test_answers: The list of test answers.
    :param embedding_model: The name of the embedding model.
    :param generator_model: The name of the generator model.
    :param temperature: The temperature setting for the model.
    :param repeat_penalty: The repeat penalty setting for the model.
    """
    filename = f"./test_data/test_data.json"

    # Prepare the data
    test_data = {
        'test_questions': test_questions, 
        'test_answers': test_answers
    }

    # Make directory if it does not exist and save the file
    os.makedirs(os.path.dirname(filename), exist_ok= True)
    with open(filename, 'w') as f: 
        json.dump(test_data, f, indent = 4)
    
    print(f'Test data saved as {filename}')

def run_evaluation_for_models(embedding_models: list, generator_models: list, temperatures: list[float], repeat_penalties: list[float], similarity_function:str, prompt: str, cut_question=True, overwrite=False, test=False):
    """
    Perform a parameter search on the pipeline. Save the results as csv after every run inside results. Skips failed runs, logs erro and continues.

    params documented in main
    """
    # nested for loop to try all combinations
    for embedding_model in embedding_models:
        for generator_model in generator_models:
            for temperature in temperatures:
                for repeat_penalty in repeat_penalties:
                    # try the run to not panic on failed runs
                    try:
                        # Skip existing result if conditions are met
                        if not overwrite and check_if_results_exist(embedding_model, generator_model, temperature, repeat_penalty, similarity_function):
                            print(f"Results already exist for {embedding_model} and {generator_model}. Skipping...")
                            continue

                        # Embed documents only if results do not exist
                        doc_store_name = f"DocMerged_{embedding_model.replace('/', '_')}.json"
                        if os.path.exists(f'./data/{doc_store_name}'):
                            print(f"Document store {doc_store_name} already exists. Skipping embedding step.")
                        else: 
                            filtered_docs = embed_documents_grouped(embedding_model= embedding_model)
                            save_database_to_disk(filtered_docs, path='./data', name=doc_store_name)
                        
                        # create rag pipeline according to params
                        rag_pipeline = create_pipeline(embedding_model, generator_model, doc_store_name = doc_store_name, temperature = temperature, prompt = prompt, repeat_penalty= repeat_penalty, cut_question=cut_question, similarity_function=similarity_function)
                        
                        # load all questions answers and documents
                        questions = load_json_file('data/querys.json')
                        ground_truth_answers = load_json_file('data/answers.json')
                        all_documents = extract_document_contents(f'./data/{doc_store_name}')
                        
                        # map documents to their corresponding row
                        with open("./data/doc_lookup.json", "r") as f:
                            lookup_table = json.load(f)
                        ground_truth_docs = [all_documents[lookup_table.get(str(i))] for i in range(len(questions))]

                        # split the data into validation 80% and testing 20% (training not required)
                        questions, test_questions = split_list_data(questions, val_ratio= 0.8, test_ratio= 0.2)
                        ground_truth_answers, test_answers = split_list_data(ground_truth_answers, val_ratio = 0.8, test_ratio = 0.2)
                        ground_truth_docs, test_docs = split_list_data(ground_truth_docs, val_ratio = 0.8, test_ratio = 0.2)
                        if not test:
                            # save test data just in caes
                            save_test_data(test_questions, test_answers)
                        else:
                            questions = test_questions
                            ground_truth_answers = test_answers
                            ground_truth_docs = test_docs
                        
                        # get evaluation results
                        results_df = evaluate_pipeline(rag_pipeline, questions, ground_truth_answers, ground_truth_docs)
                        
                        # save evaluation results
                        save_evaluation_results(results_df, embedding_model, generator_model, temperature, repeat_penalty, similarity_function)
                    except Exception as e:
                        # If the run fails log the errors
                        error_message = f"Error occurred for {embedding_model} and {generator_model}: {traceback.format_exc(e)}"
                        print(error_message)

                        log_error_to_file(error_message)

if __name__ == "__main__":
    """
    Do a hyperparameter search on the entire pipeline. Parameters can be specified below:
        embedding_models: list(str) - path to embedding model (haystack and huggingface paths are supported)
        generator models: list(str) - path to local weights file
        prompt: str - special template string for LLM
        ts: list(number) - list of temperatures to try (0 no randomness, 2< a lot of randomness)
        rps: list(number) - list of repeat penalties to try (1 is no penalty, 2< a lot of penalty)
        similarity function: "dot_product" or "cosine" - similarity used for the retrieval
        cut_question: bool
        overwrite: bool
        test: bool
    """

    # Embedding models to try. Models are downloaded automatically from huggingface
    embedding_models = [
        # default: "sentence-transformers/all-MiniLM-L6-v2",
        # "multi-qa-mpnet-base-cos-v1", # (mean, 420MB)
        #"all-mpnet-base-v2", # performance (mean, 420MB)
        #"multi-qa-MiniLM-L6-cos-v1", # (mean, 80MB, better at sentence embedding)
        #"all-MiniLM-L12-v2", # (mean, 120MB, better at semantic search)
        "multi-qa-mpnet-base-dot-v1", # similarity(CLS pooling, 420MB)
        #"sentence-transformers/nli-bert-base-max-pooling", # max pooling
    ]
    
    # List of LLMs to try. Weights should be downloaded manually in gguff format.
    generator_models = [
        "model_weights/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
    ]
    
    # The LLM prompt. Parts inside {} are filled automatically
    prompt = """Given a context, provide ONLY the answers to the questions without repeating the question. 
    Keep the answers very short, only limiting yourself to directly answering the question.  
    DO NOT:
    - Generate multiple questions and answers.
    - Answer in multiple sentences.
    - Include irrelevant information.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer: """

    # list of temperatures to try
    ts = [0.6]
    # list of repeat penaltys to try
    rps = [1]
    
    similarity_function = "dot_product"
    # similarity_function = "cosine"

    # whether to cut halluctionations starting with "Question:" from the LLM answer
    cut_question = True
    # whether to overwrite results with the same name
    overwrite = False
    # whether to use test set
    test = True

    # run the hyperparameter optimization with the specified values
    run_evaluation_for_models(embedding_models, generator_models, temperatures=ts, similarity_function=similarity_function, prompt = prompt, repeat_penalties=rps, cut_question=True, overwrite=False, test=False)  