# This File Converts the videos in the "videos" folder to mp3 format 

import os
import subprocess

files = os.listdir("videos")
print(files)

for file in files:
    # print(file)
    tutorial_no = file.split(" [")[0].split(" #")[1]
    file_name = file.split(" ｜ ")[0]
    print(tutorial_no, file_name)
    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{tutorial_no}_{file_name}.mp3"])