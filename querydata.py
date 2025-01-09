import pandas as pd
df_all = pd.read_parquet('./data/test-00000-of-00002.parquet', engine='pyarrow')
query_list = df_all['query'].loc[0:228].tolist()
answer_list = df_all['answer'].loc[0:228].tolist()
chunk_id_list = df_all['chunk_id'].loc[0:228].tolist()
text_list = df_all['text_description'].loc[0:228].tolist()

query_dict = {}

for query, text in zip(query_list, text_list):
    if query not in query_dict:
        query_dict[query] = []
    query_dict[query].append(text)

# [[query1, [id1, id2]], [query2, [id3, id4, id5]], ...]
result = [[query, text] for query, text in query_dict.items()]

print(result[0])