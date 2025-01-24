"""
Run this first to preprocess the raw data
"""

import os
import json
import pandas as pd

# read raw data (00002 is the only one we need to collect question, answer and text)
df_all = pd.read_parquet(
    './data/test-00000-of-00002.parquet', engine='pyarrow')

# merge text chunks belonging to the same document into one chunk, duplicate documents are present
df_all["document"] = df_all.groupby(
    "query")["text_description"].transform(lambda x: '\n'.join(x))
columns_of_interest = ["query", "answer", "document"]

# new dataframe by deduplicating the documents
df_unique = df_all.drop_duplicates(
    subset=columns_of_interest).dropna(subset=columns_of_interest).reset_index(drop=True)

# Map indices in the original dataframe (that includes duplications) to the indices in the deduplicated dataframe
# This way we obtain tha mapping: Original index (same as for querys and answers) -> Deduplicated index (in document store)
doc_lookup = {dup_idx: pd.Index(df_unique.drop_duplicates(subset="document")[
    "document"]).get_loc(value) for dup_idx, value in enumerate(df_unique["document"])}

# convert the columns to lists
querys = df_unique['query'].tolist()
answers = df_unique['answer'].tolist()
texts = df_unique['document'].tolist()

# Ensure the data folder exists
os.makedirs('./data', exist_ok=True)

# Save each list in separate files
with open('./data/querys.json', 'w', encoding='utf-8') as f:
    json.dump(querys, f, ensure_ascii=False, indent=4)
with open('./data/texts.json', 'w', encoding='utf-8') as f:
    json.dump(texts, f, ensure_ascii=False, indent=4)
with open('./data/answers.json', 'w', encoding='utf-8') as f:
    json.dump(answers, f, ensure_ascii=False, indent=4)

# Also save the lookup dictionary
with open('./data/doc_lookup.json', 'w', encoding='utf-8') as f:
    json.dump(doc_lookup, f)

print("Data saved in separate files: querys.json, texts.json, answers.json and doc_lookup.json")
