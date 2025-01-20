import os
import pandas as pd
from haystack import Pipeline
from haystack.components.evaluators.sas_evaluator import SASEvaluator
from haystack.evaluation.eval_run_result import EvaluationRunResult

# Define the evaluation function
def evaluate_pipeline(predicted_answers, ground_truth_answers):
    # Initialize the SAS Evaluator
    eval_pipeline = Pipeline()
    eval_pipeline.add_component("sas_evaluator", SASEvaluator(model="sentence-transformers/all-MiniLM-L6-v2"))
    
    # Prepare the inputs for the evaluation
    inputs = {
        "predicted_answers": predicted_answers,
        "ground_truth_answers": list(ground_truth_answers)
    }

    # Evaluate the predicted answers against the ground truth answers
    results = eval_pipeline.run({"sas_evaluator": inputs})
    
    # Create an EvaluationRunResult and generate a score report
    evaluation_result = EvaluationRunResult(run_name="sas_evaluation_run", inputs=inputs, results=results)
    evaluation_result.score_report()  # Optionally, this can be printed or logged
    
    # Convert the results to a pandas DataFrame and return it
    results_df = evaluation_result.to_pandas()
    return results_df

# Define the path to the 'results' folder
results_path = "results"

# Ensure the folder exists
if not os.path.exists(results_path):
    print(f"Directory '{results_path}' does not exist.")
    exit()

# Process each file in the results folder
for filename in os.listdir(results_path):
    # Check if the file is a CSV
    if filename.endswith(".csv"):
        filepath = os.path.join(results_path, filename)
        
        try:
            # Read the CSV file
            df = pd.read_csv(filepath)
            
            # Check for necessary columns
            if 'predicted_answer' in df.columns and 'answer' in df.columns:
                # Update the 'predicted_answer' column
                df['predicted_answer'] = df['predicted_answer'].str.split('Question:').str[0].str.strip()
                
                # Evaluate the updated predictions
                rag_answers = df['predicted_answer'].tolist()
                ground_truth_answers = df['answer'].tolist()
                evaluation_scores_df = evaluate_pipeline(rag_answers, ground_truth_answers)
                
                # Check if the evaluation returned a valid DataFrame
                if not evaluation_scores_df.empty:
                    # Add the 'sas_evaluator' column to the original DataFrame with the results
                    df['sas_evaluator'] = evaluation_scores_df['sas_evaluator'].tolist()
                
                    # Save the modified DataFrame back to the same file
                    df.to_csv(os.path.join("new_results", filename), index=False)
                    print(f"Processed and saved: {filename}")
                else:
                    print(f"No valid evaluation results for {filename}. Skipping file.")
            else:
                print(f"Required columns ('predicted_answer', 'answer') not found in {filename}. Skipping file.")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")