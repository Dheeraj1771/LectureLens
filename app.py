import os 
import json
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from groq import Groq
from pinecone import Pinecone

# Get The Hugging Face Token from .env File
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize the Hugging Face Client
if HF_TOKEN:
    hf_client = InferenceClient(token=HF_TOKEN)
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

course_catalog = {
    "7 AI Terms You Need to Know: Agents, RAG, ASI & More": "VSFuqMh4hus",
    "Is RAG Still Needed? Choosing the Best Approach for LLMs": "UabBYexBD4k",
    "What Is an AI Stack? LLMs, RAG, & AI Hardware":"RRKwmeyIc24",
    "AI Periodic Table Explained: Mapping LLMs, RAG & AI Agent Frameworks":"ESBMgZHzfG0",
    "RAG vs Agentic AI: How LLMs Connect Data for Smarter AI":"fB2JQXEH_94"
}

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

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_video_id' not in st.session_state:
    st.session_state.current_video_id = None
if 'video_start_time' not in st.session_state:
    st.session_state.video_start_time = 0
     
# Side Bar Setup
with st.sidebar:
    st.header("📖 Course Material")
    st.markdown("Select a pre-indexed video from the catalog to ask questions about it.")
    
    available_videos = ["Select a video..."] + list(course_catalog.keys())
    selected_video = st.selectbox("Choose a Lecture:", available_videos)
    
    if selected_video != "Select a video...":
        st.session_state.current_video_id = course_catalog[selected_video]
    else:
        st.session_state.current_video_id = None
    
    if st.session_state.current_video_id:
        st.divider()
        st.write("📺 **Video Player**")
        youtube_url = f"https://www.youtube.com/watch?v={st.session_state.current_video_id}"
        st.video(youtube_url, start_time=int(st.session_state.video_start_time))
    
# Main Page
st.title("🎓 LectureLens - A RAG Based AI Teaching Assistant")
st.subheader("What would you like to learn from this lecture?")

# To keep chat content in place
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input("E.g., What is RAG?"):
    # Block user from asking question is no video is select
    if not st.session_state.current_video_id:
        st.error("Please select a video from the sidebar first!")
        st.stop()
        
    # Display the User's Question
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display the AI's Processing and Answer State
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🧠 Searching course materials...")
        
        try:
            # Embed the User's Question and convert it to list
            question_embedding = create_embedding([prompt])[0].tolist()
            
            # 4. Query Pinecone to get top 5 chunks, with metadata filtering
            query_result = pinecone_index.query(
                vector=question_embedding,
                top_k=5,
                include_metadata=True,
                filter={
                    "video_id": {"$eq": st.session_state.current_video_id}
                }
            )
            
            # Extract the Metadata from the top5 chunks
            retrieved_chunks = []
            for match in query_result['matches']:
                meta = match['metadata']
                minutes = int(meta['start'] // 60)
                seconds = int(meta['start'] % 60)
                meta['formatted_time'] = f"{minutes}:{seconds:02d}"
                retrieved_chunks.append(meta)
            
            # Auto seek the video player to the top result's start
            if retrieved_chunks:
                top_match_time = retrieved_chunks[0]['start']
                st.session_state.video_start_time = int(top_match_time)
                
            # Convert the context into string for LLM to understand
            context_json = json.dumps(retrieved_chunks)
            message_placeholder.markdown("🤖 Synthesizing Answer...")
            
            # Build the Dynamic Prompt
            system_prompt = f'''You are an AI teaching assistant for an educational video course. 
            Here are the most relevant video subtitle chunks based on the user's query. They contain the video title, the timestamp (formatted as MM:SS), and the spoken text:

            {context_json}
            -----------------------------------------------------------------------
            User's Question: "{prompt}"

            Based ONLY on the provided video chunks, answer the user's question in a natural, helpful, human format. 
            Crucially, you must tell the user exactly which video title and at what timestamp they can find this information (use the 'formatted_time' field provided). 
            Do not mention the JSON data format, the word "chunks", or the raw seconds directly to the user. 
            If the user asks a question that cannot be answered using the provided video materials, politely inform them that you can only answer questions related to the uploaded course content.
            '''
            
            # Call Llama 3.3 via Groq
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful Teaching Assistant."},
                    {"role": "user", "content": system_prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
            )
            
            # Display the final answer
            full_response = chat_completion.choices[0].message.content
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            st.rerun()
            
        except Exception as e:
            message_placeholder.markdown(f"**Error:** {str(e)}")
