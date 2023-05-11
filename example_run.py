#example file of processing the data 
import pickle 
import pandas 
import pandas as pd
import numpy as np
import os 
import pickle 
import random
from collections import defaultdict
from feat import Detector 
from extract_data import *

#load original annotations
with open('/mnt/swordfish-pool2/ccu/as5957-cache.pkl', 'rb') as handle:
    annotations = pickle.load(handle)
    
# filter and save video only annotations
video_annotations = filter_only_video(annotations)
with open("./video_annotations.pkl", 'wb') as f:
        pickle.dump(video_annotations, f)
    
    
# get changepoints a video file and add as a list 
change_points = defaultdict(list)
for file_id in annotations:
    changes = get_change_points(annotations,file_id)
    for c in changes:
        change_points[file_id].append(c['timestamp'])

with open("./video_changepoints.pkl", 'wb') as f:
        pickle.dump(change_points, f)

#    {'file_id': [timestamps of change points]}
with open("./video_changepoints.pkl", 'rb') as handle:
    video_change_points = pickle.load(handle)
    
#get change_point features
detector = Detector()
change_point_features_train = extract(video_change_points,annotations,detector,'train')
with open("./change_point_features_train.pkl", 'wb') as f:
    pickle.dump(change_point_features_train, f)


# get non change points 
non_change_points_dict = get_non_change_points(annotations, video_change_points)

non_change_point_features_train = extract(non_change_points_dict,annotations,detector,'train')
with open("./non_change_point_features_train.pkl", 'wb') as f:
    pickle.dump(non_change_point_features_train, f)

#extract data for the validation set 
dyadic_val_set = extract_val_set(annotations, detector,'val')
with open("./val_set.pkl", 'wb') as f:
    pickle.dump(dyadic_val_set, f)
    
   
# load frames and get similarities
with open("./non_change_point_features_train.pkl", 'rb') as handle:
    non_change_points_frames = pickle.load(handle)

with open("./change_point_features_train.pkl", 'rb') as handle:
    change_points_frames = pickle.load(handle)

#change 'head_pose' to whichever feature similarity is being detected for ('all', 'emotion', 'musc', 'head_pose")

sims_dict_cp, sims_df_cp = similarity_scores(change_points_frames,'all', 'cp')
sims_dict_ncp, sims_df_ncp = similarity_scores(non_change_points_frames,'all', 'ncp')

#save the simiarlites in a df so that later they can be uploaded for graphing and analysis 
with open("./non_change_point_sims_all.pkl", 'wb') as f:
    pickle.dump(sims_dict_ncp, f)

with open("./change_point_sims_all.pkl", 'wb') as f:
    pickle.dump(sims_dict_cp, f)

sims_df_cp.to_csv("./change_point_sims_all.csv")
sims_df_ncp.to_csv("./nonchange_point_sims_all.csv") #note these do not have impact scalars appended 


time_stamp_scalars = get_impact_scalars(annotations, list(sims_dict_cp['file_id']))

change_point_all_sims = append_impact_scalars(sims_df_cp, time_stamp_scalars)
change_point_all_sims.to_csv("./change_point_all_sims_scalars.csv",index=False) #append impact scalars to similiarty df 
#repeat for all features we want similarities for 

#to get impact scalars for our extracted changepoint data load the pickle file and pass the file_ids we want impact scalars of 

time_stamp_scalars_train = get_impact_scalars(annotations, list(change_point_features_train.keys()))
   
with open("/home/as5957/research_spring_2023/train_time_scalars.pkl", 'wb') as f:
    pickle.dump(time_stamp_scalars_train, f)

#to get validation data 
#first process cps
change_point_features_val = extract(change_points,annotations,detector,'val')
with open("./cp_features_val.pkl", 'wb') as f:
    pickle.dump(change_point_features_val, f)
#run this through filter.py for dyadic videos only bc we only want ncps for dyadic videos as well

#change the file to file_name for dyadic videos (this will take a while)
with open('/home/as5957/research_spring_2023/cp_features_val_dyadic.pkl', 'rb') as handle:
    cp = pickle.load(handle)

ncp = {}
count = 0
for file_id in cp.keys():
    print(count)
    count += 1
    
    start = annotations[file_id]['start']
    end = annotations[file_id]['end']
    utterances = annotations[file_id]['utterances'] #list of dictionaries with vars for an utterance 
    print(len(utterances))
    directory = annotations[file_id]['processed_dir']
    ncp[file_id] = {}
    for utterance in utterances:
        
        videos = utterance['video_frames']
        span = []
        times = []
        if len(videos) > 0: 
            new_videos = [videos[0],videos[-1]]
            for (time,frame) in new_videos:
                if start <= time and end >= time:
                    times.append(time)
                    frame_path = os.path.join(directory, frame)
                    df = detector.detect_image(frame_path)
                    span.append(df)
            if len(span) > 0 and all(len(d) == 2 for d in span):
                 ncp[file_id][frozenset(times)] = span

with open("./ncp_val.pkl", 'wb') as f:
    pickle.dump(ncp, f)
