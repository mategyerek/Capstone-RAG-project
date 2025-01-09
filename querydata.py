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