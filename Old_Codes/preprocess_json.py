# This Program is used to read the chunks of text and create embeddings for them using the BGE-M3 model. The embeddings are then stored in a Pandas DataFrame for later retrieval.
import os
import json
import requests
import pandas as pd
import joblib


def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model" : "bge-m3",
        "input" : text_list
    })

    embedding = r.json()["embeddings"]
    return embedding

jsons = os.listdir('newjsons') # List all the json files in the jsons directory
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f'newjsons/{json_file}') as f:
        content = json.load(f)
    
    print(f"Creating embeddings for {json_file} with {len(content['chunks'])} chunks")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
    
    for i, chunk in enumerate(content['chunks']):
        chunk["chunk_id"] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)
    
df = pd.DataFrame.from_records(my_dicts)

# Save this DataFrame
joblib.dump(df, 'embeddings.joblib')

