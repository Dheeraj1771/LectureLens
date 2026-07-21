import os 
import re
import json
import pandas as pd 
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi
from pinecone import Pinecone

# Get The Hugging Face Token from .env File
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize the Hugging Face Client
if HF_TOKEN:
    hf_client = InferenceClient(HF_TOKEN)
else:
    st.error("Missing HF_TOKEN in .env File")

# Initialize the Groq API Client
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    st.error("Missing GROQ_API_KEY in .env File")

# Initialize the Pinecone API Client
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("lecturelens")
else:
    st.error("Missing PINECONE_API_KEY in .env File")


    
def extract_video_id(url):
    # Extract 11 Character Video ID from the URL
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    else:
        return None

def create_embedding(text_list):
    # Send the Text to Hugging Face Inference API to get BGE-M3 and convert the text into embeddings
    try:
        response = hf_client.feature_extraction(
            text_list,
            model="BAAI/bge-m3"
        )
        return np.array(response)
    except Exception as e:
        raise Exception(f"Hugging Face API Error {str(e)}")
    
# Page Setup
st.set_page_config(
    page_title="LectureLens - A RAG Bases AI Teaching Assistant",
    page_icon="🎓",
    layout="wide"
)

# Side Bar Setup
with st.sidebar:
    st.header("📖 Course Material")
    st.markdown("Paste The YouTube Link below to add it to the AI's Knowledge Base...")
    
    youtube_url = st.text_input("YouTube URL: ")
    video_title = st.text_input("Video Title (E.g. Introduction to Data Science)")
    video_number = st.number_input("Video Number: ", min_value=1, step=1)
    
    if st.button("Process Video"):
        if youtube_url and video_title:
            video_id = extract_video_id(youtube_url)
            
            if not video_id:
                st.error("Invalid YouTube URL")
            else:
                with st.status("Processing Video...", expanded=True) as status:
                    try:
                        st.write("1. Retrieving video transcripts...")
                        ytt_api = YouTubeTranscriptApi()
                        transcript_list = ytt_api.list(video_id)
                        
                        # Grab the first available transcript track
                        transcript = next(iter(transcript_list))
                        
                        st.write(f"   - Found transcript in language: '{transcript.language}'")
                        
                        # If Transcript is in English
                        if transcript.language_code == 'en':
                            st.write("  - Video is already in English. Fetching directly..")
                            raw_transcript = transcript.fetch()
                        else:
                            st.write("   - Automatically translating content to English...")
                            raw_transcript = transcript.translate('en').fetch()                    
                        
                        st.write("2. Chunking text into 30-second blocks...")
                        chunks = []
                        current_text = ""
                        start_time = 0
                        
                        for i, item in enumerate(raw_transcript):
                            i_start = item['start'] if isinstance(item, dict) else item.start
                            i_text = item['text'] if isinstance(item, dict) else item.text
                            i_duration = item['duration'] if isinstance(item, dict) else item.duration
                            
                            if current_text == "":
                                start_time = i_start
                            
                            current_text += i_text + ' '
                            end_time = i_start + i_duration
                            
                            if (end_time - start_time) >= 30 or i == len(raw_transcript) - 1:
                                chunks.append({
                                    'title': video_title,
                                    'number': video_number,
                                    'start': start_time,
                                    'end': end_time,
                                    'text': current_text.strip().replace('\n', ' ')
                                })
                                current_text = ""
                        df = pd.DataFrame(chunks)
                        
                        st.write(f"3. Generating Embeddings for {len(df)} chunks...")
                        text_to_embed = df['text'].tolist()
                        embeddings_matrix = create_embedding(text_to_embed)
                        df['embeddings'] = list(embeddings_matrix)
                        
                        st.write("4. Saving to Pinecone Cloud Database...")
                        vectors_to_upsert = []
                        for idx, row in df.iterrows():
                            # Create a unique ID so if a user uploads the same video twice, it overwrites instead of duplicates!
                            chunk_id = f"{video_id}_{row['start']}"
                            
                            # Convert the numpy array to python list for pinecone db
                            vectror_values = row['embeddings'].tolist()
                            
                            # Attach Human Readable text as metadata
                            metadata = {
                                "title": row['title'], 
                                "number": row['number'],
                                "start": row['start'],
                                "end": row['end'],
                                "text": row['text']
                            }
                            vectors_to_upsert.append((chunk_id, vectror_values, metadata))
                       
                        # upload to the cloud in batches
                        pinecone_index.upsert(vectors=vectors_to_upsert)
                        
                        status.update(label="Completed! Video Added to Cloud.", state="complete", expanded=False)
                        st.success(f"Successfully Processed, Translated, and Embedded {len(df)} chunks from '{video_title}'")
                        
                    except Exception as e:
                       status.update(label="Error Processing Video", state='error', expanded=False)
                       st.error(f"Failed: {e}")
        else:
            st.warning("Please provide both a URL and a Title.")          
    
# Main Page
st.title("🎓 LectureLens - A RAG Based AI Teaching Assistant")
st.subheader("What would you like to learn from this lecture?")

# To keep chat content in place
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input("E.g., Where is Data Science Basics taught in this course?"):
    # 1. Display the User's Question
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Display the AI's Processing and Answer State
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🧠 Searching course materials...")
        
        try:
            # # 3. Load the Dataset
            # df = joblib.load('embedding.joblib')
            
            # 3. Embed the User's Question and convert it to list
            question_embedding = create_embedding([prompt])[0].tolist()
            
            # 4. Query Pinecone to get top 5 chunks
            query_result = pinecone_index.query(
                vector=question_embedding,
                top_k=5,
                include_metadata=True
            )
            
            # 5. Extract the Metadata from the top5 chunks
            retrieved_chunks = [match['metadata'] for match in query_result['matches']]
            # Convert the context into string for LLM to understand
            context_json = json.dumps(retrieved_chunks)
            message_placeholder.markdown("🤖 Synthesizing Answer...")
            
            # 6. Build the Dynamic Prompt
            system_prompt = f'''You are an AI teaching assistant for an educational video course. 
            Here are the most relevant video subtitle chunks based on the user's query. They contain the video title, video number, start time (seconds), end time (seconds), and the spoken text:

            {context_json}
            -----------------------------------------------------------------------
            User's Question: "{prompt}"

            Based ONLY on the provided video chunks, answer the user's question in a natural, helpful, human format. 
            Crucially, you must tell the user exactly which video (title/number) and at what timestamp they can find this information. 
            Do not mention the JSON data format or the word "chunks" directly to the user. 
            If the user asks a question that cannot be answered using the provided video materials, politely inform them that you can only answer questions related to the uploaded course content.
            '''
            
            # 7. Call Llama 3.3 via Groq
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful teaching assistant."},
                    {"role": "user", "content": system_prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
            )
            
            # 8. Display the final answer
            full_response = chat_completion.choices[0].message.content
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            message_placeholder.markdown(f"**Error:** {str(e)}\n\n(Did you upload a video yet?)")
            
