import pandas as pd
df_all = pd.read_parquet('./data/test-00000-of-00002.parquet', engine='pyarrow')
query_list = df_all['query'].loc[0:228].tolist()
answer_list = df_all['answer'].loc[0:228].tolist()
chunk_id_list = df_all['chunk_id'].loc[0:228].tolist()
text_list = df_all['text_description'].loc[0:228].tolist()

# list of texts
texts = []
current_query = query_list[0]
current_text = []

for query, text in zip(query_list, text_list):
    if query == current_query:
        current_text.append(text)
    else:
        texts.append(current_text)
        current_query = query
        current_text = [text]

if current_text:
    texts.append(current_text)

# list of answers
answers = [answer_list[0]]
for answer in answer_list[1:]:
    if answer != answers[-1]:
        answers.append(answer)

# list of querys
querys = [query_list[0]]
for query in query_list[1:]:
    if query != querys[-1]:
        querys.append(query)

i = 1
print(querys[i], texts[i], answers[i])

import json
import os

# Ensure the data folder exists
os.makedirs('./data', exist_ok=True)

# Save each list in separate files
with open('./data/querys.json', 'w', encoding='utf-8') as f:
    json.dump(querys, f, ensure_ascii=False, indent=4)

with open('./data/texts.json', 'w', encoding='utf-8') as f:
    json.dump(texts, f, ensure_ascii=False, indent=4)

with open('./data/answers.json', 'w', encoding='utf-8') as f:
    json.dump(answers, f, ensure_ascii=False, indent=4)

print("Data saved in separate files: querys.json, texts.json, answers.json")


