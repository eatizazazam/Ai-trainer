import pandas as pd
import os

# input_folder ="landmarks/back_angle/Correct"
# input_folder ="landmarks/back_angle/Rounded"
# input_folder ="landmarks/knee_valgus/correct"
input_folder ="landmarks/knee_valgus/knee_cave"



for videos in os.listdir(input_folder):
     if videos.endswith(".csv"):
         video_path=os.path.join(input_folder,videos)
         df=pd.read_csv(video_path)
         print(f"successfully Read, {video_path}")
         print(df.isna().sum())   
 