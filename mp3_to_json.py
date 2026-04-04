# This Program is used to create chunks of text from the audio files using the Whisper model. The chunks are then stored in a JSON file for later retrieval. Each chunk contains the start and end time of the segment, the text of the segment, and the title and number of the audio file it belongs to.
import whisper
import json
import os

# Load the Whisper Model
model = whisper.load_model('large-v2')

# Get the list of audio files in the 'audios' directory
audios = os.listdir('audios')

for audio in audios:
    if "_" in audio:
        number = audio.split('_')[0]
        title = audio.split('_')[1][:-4]  # Remove .mp3 extension
        print(number, title)
        result = model.transcribe(audio=f'audios/{audio}', 
                                  language='hi',
                                  task='translate',
                                  word_timestamps=False)
        
        chunks=[]
        for segment in result["segments"]:
            chunks.append({"number": number, "title" : title, "start" : segment["start"], "end" : segment["end"], "text" : segment["text"]})
            
        chunks_with_metadata = {"chunks" : chunks, "text" : result["text"]}
        
        with open(f'jsons/{audio}.json', 'w') as f:
            json.dump(chunks_with_metadata, f, indent=4)  