import os
import numpy as np
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from pinecone import Pinecone
from huggingface_hub import InferenceClient

# Load environemt variables from .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Hugging Face Client
hf_client = InferenceClient(token=HF_TOKEN)

# Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'lecturelens'

# Connect to Pinecone index
index = pc.Index(index_name)

# IMB Video IDs
videos = [
    {"id": "VSFuqMh4hus", "title": "7 AI Terms You Need to Know: Agents, RAG, ASI & More"},
    {"id": "UabBYexBD4k", "title": "Is RAG Still Needed? Choosing the Best Approach for LLMs"},
    {"id": "RRKwmeyIc24", "title": "What Is an AI Stack? LLMs, RAG, & AI Hardware"},
    {"id": "ESBMgZHzfG0", "title": "AI Periodic Table Explained: Mapping LLMs, RAG & AI Agent Frameworks"},
    {"id": "fB2JQXEH_94", "title": "RAG vs Agentic AI: How LLMs Connect Data for Smarter AI"}
]

def chunk_transcript(transcript, chunk_size=7):
    # Groups 7 lines into one chunk
    chunks = []
    
    for i in range(0, len(transcript), chunk_size):
        # Create a group of size 7
        group = transcript[i : i + chunk_size]
        
        # Combine the text of all lines in this group
        combined_text = " ".join([item.text for item in group])
        
        # Start Time, will be the start of the first chunk in the group
        start_time = group[0].start
        
        # End Time, will be the start + duration of the last chunk in the group
        end_time = group[-1].start + group[-1].duration
        
        # Append each chunk to the chunks list
        chunks.append({
            'text': combined_text.replace("\n", " "),
            'start': start_time,
            'end': end_time
        })
    return chunks

def create_embeddings(text_list):
    # Send list of text chunks to Hugging Face to get the BGE-M3 embeddings (1024 dimensions)
    try:
        response = hf_client.feature_extraction(
            text_list,
            model="BAAI/bge-m3"
        )
        # Convert to a list of floats so Pinecone can ingest it seamlessly
        return np.array(response).tolist()
    except Exception as e:
        raise Exception(f"Hugging Face API Error: {e}")  
    
# Main function to build the database
def main():
    print("Starting the process of building the database...")
    
    yt_api = YouTubeTranscriptApi()
    
    for video in videos:
        print(f"Processing video: {video['title']}...")
        
        # Extract Youtube Video Transcript
        try:
            raw_transcript = yt_api.fetch(video['id'])
            print(f" - Downloaded {len(raw_transcript)} raw transcript lines")
        except Exception as e:
            print(f" - Error fetching transcript for {video['title']} , with ID {video['id']}: \n{e}")
            continue

        # Create chunks of the transcript
        chunks = chunk_transcript(raw_transcript, chunk_size=7)
        print(f" - Grouped the transcript into {len(chunks)} sematic chunks")
        
        # Embed and Uprest in Batchesm, to avoid overloading API
        batch_size = 30
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            
            # Extract just the text
            text_batch = [item['text'] for item in batch]
            
            print(f" - Embedding batch {i} to {i + len(batch)}...")
            embeddings = create_embeddings(text_batch)
            
            # Prepare vectors for Pinecone
            pinecone_vactors = []
            
            for j, chunk in enumerate(batch):
                # Create unique ID for each chunk
                chunk_id = f"{video['id']}_chunk_{i+j}"
                
                # Bundle up the vectors and metadata together
                pinecone_vactors.append({
                    'id': chunk_id,
                    'values': embeddings[j],
                    'metadata': {
                        'video_id': video['id'],
                        'title': video['title'],
                        'start': chunk['start'],
                        'end': chunk['end'],
                        'text': chunk['text']
                    }
                })
            # Push the batch to Pinecone db
            index.upsert(vectors=pinecone_vactors)
            print(" - Upserted Batch to Pinecone db.")
    print("Ingestion completed!! Database is ready")

if __name__ == '__main__':
    main()
