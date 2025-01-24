import os
import json
from haystack import Pipeline
from haystack.evaluation.eval_run_result import EvaluationRunResult
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.components.generators.chat import OpenAIChatGenerator
import os
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder, AnswerBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.evaluators.faithfulness import FaithfulnessEvaluator
from haystack.components.evaluators.sas_evaluator import SASEvaluator
from embed_document import load_json_file, extract_document_contents, embed_documents_grouped, save_database_to_disk
import logging
import transformers
import pickle 
import pandas as pd 
from bert_score import score 
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)


if __name__ == "__main__":
    """
    Script for evaluating ChatGPT4o-mini's performance without RAG. 

    This script loads document data, queries, and ground truth answers, sets up a pipeline with text embedding,
    chat generation, and answer building, and evaluates the pipeline's performance using various metrics.

    Key Functions:
    - `evaluate_pipeline`: Evaluates the pipeline's performance by generating answers to the questions,
      comparing them to ground truth answers, and computing metrics such as SAS and BERTScore.
    - `save_evaluation_results_norag`: Saves evaluation results to a CSV file.

    Components:
    - Embedding model: "sentence-transformers/all-MiniLM-L6-v2"
    - Generator model: "gpt-4o-mini"

    :return: The evaluation results are saved to a CSV file with metrics for pipeline performance.
    """

    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    text_embedder = SentenceTransformersTextEmbedder(
        model=embedding_model)


    template = [ChatMessage.from_user("""Provide ONLY the answers to the questions without repeating the question. 
    Keep the answers very short, only limiting yourself to directly answering the question.  
    DO NOT:
    - Generate multiple questions and answers.
    - Answer in multiple sentences.
    - Include irrelevant information.


    Question: {{question}}
    Answer: """)]


    prompt_builder = ChatPromptBuilder(template=template)

    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "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"
    chat_generator = OpenAIChatGenerator(model="gpt-4o-mini")

    basic_rag_pipeline = Pipeline()
    # Add components to the pipeline
    basic_rag_pipeline.add_component("text_embedder", text_embedder)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("generator", chat_generator)
    basic_rag_pipeline.add_component("answer_builder", AnswerBuilder())

    # Now, connect the components to each other
    basic_rag_pipeline.connect("prompt_builder.prompt", "generator.messages")
    basic_rag_pipeline.connect("generator.replies", "answer_builder.replies")


    # Loading the necessary data
    questions = load_json_file('data/querys.json')
    ground_truth_answers = load_json_file('data/answers.json')
    all_documents = extract_document_contents('./data/DocMerged.json')
    with open("./data/doc_lookup.json", "r") as f:
        lookup_table = json.load(f)
    ground_truth_docs = [all_documents[lookup_table.get(
        str(i))] for i in range(len(questions))]

    def evaluate_pipeline(rag_pipeline, questions, ground_truth_answers):
        """
        Evaluate the pipeline using several metrics.

        :param rag_pipeline: The pipeline to evaluate.
        :param questions: The list of questions.
        :param ground_truth_answers: The ground truth answers for evaluation.
        :param ground_truth_docs: The ground truth documents used for retrieval.
        :return: Evaluation results in a pandas DataFrame.
        """
        eval_pipeline = Pipeline()
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

        results = eval_pipeline.run(
            {
                "sas_evaluator": {"predicted_answers": rag_answers, "ground_truth_answers": list(ground_truth_answers)},
            }
        )

        inputs = {
            "question": list(questions),
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

    def save_evaluation_results_norag(results_df, generator_model):
        """
        Save the evaluation results as a CSV file.

        :param results_df: The evaluation results DataFrame.
        :param embedding_model: The name of the embedding model.
        :param generator_model: The name of the generator model.
        """
        filename = f"./data/results_{generator_model})_NO-RAG.csv"
        results_df.to_csv(filename, index=False)
        print(f'Evaluation results saved as {filename}')

save_evaluation_results_norag(evaluate_pipeline(basic_rag_pipeline, questions = questions, ground_truth_answers= ground_truth_answers), 'gpt4o-mini')
