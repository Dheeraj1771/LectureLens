import joblib
import requests
# import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model" : "bge-m3",
        "input" : text_list
    })

    embedding = r.json()["embeddings"]
    return embedding

def inference(prompt):
    print("Thinking...")
    r = requests.post("http://localhost:11434/api/generate", json={
        "model" : "llama3.2",
        "prompt" : prompt,
        "stream": False
    })
    
    response = r.json()
    # print(response)
    return response

df = joblib.load('embeddings.joblib')

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]
# print("Question Embedding: ", question_embedding)

# Calculate cosine similarity between the question embedding and the chunk embeddings (i.e. df['embedding'])

# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
# np.vstack convert the array into 2 dimensional Array

similarities = cosine_similarity(np.vstack(df['embedding'].values), [question_embedding]).flatten()
# print(similarities)
top_result = 5
max_indx = similarities.argsort()[::-1][0:top_result]
# print(max_indx)

new_df = df.loc[max_indx]
# print(new_df[['title', 'number', 'text']])

prompt = f'''I am teaching web development in my Sigma web development Here are video subtitle chunks containing video title, video numberm, start time in seconds, 
end time in seconds, the text at that time:
 
{new_df[['title', 'number', 'start', 'end', 'text']].to_json(orient='records')}
-----------------------------------------------------------------------
"{incoming_query}"
User asked this question realted to the video chunks, you have to answer in a human format (don't mention in the above format, its just for you) where and how much 
content is taught in which video (in ehich video and at what timestamp) and guide the user to go to that 
particular video. If user asks unrelated questions, tell him that you can only answer questions related to the course 
'''

with open('prompt.txt', 'w') as f:
    f.write(prompt)

response = inference(prompt)['response']
print(response)

with open('response.txt', 'w') as f:
    f.write(response)

# for index, item in new_df.iterrows():
#     print(index, item['title'], item['number'], item['text'], item['start'], item['end'])