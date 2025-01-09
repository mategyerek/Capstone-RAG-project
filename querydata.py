import pandas as pd
df_all = pd.read_parquet('./data/test-00000-of-00002.parquet', engine='pyarrow')
query_list = df_all['query'].loc[0:228].tolist()
answer_list = df_all['answer'].loc[0:228].tolist()
chunk_id_list = df_all['chunk_id'].loc[0:228].tolist()
text_list = df_all['text_description'].loc[0:228].tolist()

# querys and texts
query_dict = {}

for query, text in zip(query_list, text_list):
    if query not in query_dict:
        query_dict[query] = []
    query_dict[query].append(text)

# [[query1, [id1, id2]], [query2, [id3, id4, id5]], ...]
query_text = [[query, text] for query, text in query_dict.items()]

print(query_text[0])

# querys and answers
queryanswer = []

for i in range(len(query_list)):
    if i == 0 or [query_list[i], answer_list[i]] != queryanswer[-1]:
        # [[query1, answer1], [query2, answer2], ...]
        queryanswer.append([query_list[i], answer_list[i]])

print(queryanswer[0])