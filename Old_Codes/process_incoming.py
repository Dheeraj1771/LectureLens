# Import the Necessary Libraries
import os
import joblib
# import requests
from huggingface_hub import InferenceClient
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from groq import Groq

# 1. Load the API from the hidden .env file into the Python's environment memory
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("HF TOKEN DETECTED:", HF_TOKEN)
print("GROQ KEY DETECTED:", GROQ_API_KEY)

# 2. Initialize the Groq client so that we can make API calls
client = Groq(api_key=GROQ_API_KEY)
hf_client = InferenceClient(token=HF_TOKEN)

def create_embeddings(text_list):
    # Send the Text to Hugging Face Inference API to get BGE-M3 and convert the text into embeddings
    try:
        response = hf_client.feature_extraction(
            text_list,
            model="BAAI/bge-m3"
        )
        return np.array(response)
    except Exception as e:
        raise Exception(f"Hugging Face API Error {str(e)}")
    

def inference(prompt):
    # Call the Groq API to generate final output using Llama 3.3
    print("Groq Initializing..")
    print("Llama 3.3 Loading...")
    print("Thinking in progress...")
    
    # Standard format for chat completion of Groq
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful teaching assistant."
            }, 
            {
                "role": "user",
                "content": prompt
            }
        ], 
        model="llama-3.3-70b-versatile",
        temperature=0.3,
    )
    return chat_completion.choices[0].message.content

# Main Pipeline Execution

# Load the embeddings.joblib file into df
df = joblib.load("embeddings.joblib") 

incoming_query = input("Ask a Question: ")

# Convert user question into vector using Hugging Face
question_embedding = create_embeddings([incoming_query])[0]

# Compare the query vector to every chunk in the df
similarity = cosine_similarity(np.vstack(df["embedding"].values), [question_embedding]).flatten()

# Get the Top 5 Results
top_result = 5
max_indx = similarity.argsort()[::-1][0:top_result]
new_df = df.loc[max_indx]

# Inject the retrieved chunks into your prompt template
# Inject the retrieved chunks into your prompt template
prompt = f'''You are an AI teaching assistant for an educational video course. 
Here are the most relevant video subtitle chunks based on the user's query. They contain the video title, video number, start time (seconds), end time (seconds), and the spoken text:

{new_df[['title', 'number', 'start', 'end', 'text']].to_json(orient='records')}
-----------------------------------------------------------------------
User's Question: "{incoming_query}"

Based ONLY on the provided video chunks, answer the user's question in a natural, helpful, human format. 
Crucially, you must tell the user exactly which video (title/number) and at what timestamp they can find this information. 
Do not mention the JSON data format or the word "chunks" directly to the user. 
If the user asks a question that cannot be answered using the provided video materials, politely inform them that you can only answer questions related to the uploaded course content.
'''

# Get the response
response_text = inference(prompt)
print(response_text)

with open('response.txt', 'w') as f:
    f.write(response_text)