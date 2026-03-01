# Speech-to-Text (STT) using OpenAI's Whisper model
import whisper 
import json

# Load the pre-trained Whisper model
model = whisper.load_model("large-v2")

result = model.transcribe(audio="audios/sample.mp3",
                          language="hi",
                          task="translate",
                          word_timestamps=False)

# Extract the segments and save them in a list of dictionaries  
chunks = []
for segement in result["segments"]:
    chunks.append({"start" : segement["start"], "end" : segement["end"], "text" : segement["text"]})
print(chunks)

# Save the chunks to a JSON file
with open("output.json", "w") as f:
    json.dump(chunks, f, indent=4)

