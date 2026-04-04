# LectureLens
An AI Teaching Assistant powered by Retrieval-Augmented Generation for Lecture Timeline Intelligence

# How to Use this RAG based AI Teaching Assistant on your own data
## Step 1: Collect your videos
Move all your video files to the videos folder

## Step 2: Convert mp4 to mp3
Convert all the video files to mp3 by running video_to_mp3.py

## Step 3: Convert mp3 to json
Convert all the mp3 files to json by running mp3_to_json.py

## Step 4: Convert the json files to vectors
Use the file preprocess_json.py to convert the json files to a DataFrame with embeddings and save it as a joblib pickle

## Step 5: Prompt Generation and Feeding to LLM
Read the joblib file and load it into the memory. Then create a relevant prompt as per the user query and feed it to the LLM