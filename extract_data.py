import pandas as pd
import numpy as np
import os 
import pickle 
import random
from collections import defaultdict
from feat import Detector 


#filter annotations for only video type data 
def append_impact_scalars(df,time_scalars):
    scalars = []
    for index, row in df.iterrows():
        file_name = row['file_id']
        time_stamp = row['time_stamp']
        scalars.append(time_scalars[(file_name, time_stamp)])

    df['scalars'] = scalars

    return df 

def get_impact_scalars(annotations, file_ids):
    #return new dictionary with timescalars appended 
    c_i_s = {}
    for file_id in file_ids:
         change_points = annotations[file_id]['changepoints']
         for c in change_points:
             c_i_s[(file_id,c['timestamp'] )] = c['impact_scalar']
    return c_i_s

def filter_only_video(dictionary):
    return dict((k, dictionary[k]) for k in dictionary.keys() if dictionary[k]['data_type'] == 'video')

#returns list of change_point dictionaries for a given file_id #get the timestamp by indexing timestamp
def get_change_points(annotations,file_id):
  change_points_dict = annotations[file_id]
  change_points = change_points_dict['changepoints']
  return change_points

def process_video_frame(annotations, file_id, time):
  utterances = annotations[file_id]['utterances'] #list of dictionaries with vars for an utterance 
  for utter in utterances:
    if time <= utter['end'] and time >= utter['start']:
      return random.choice(utter['video_frames'])

def get_frames(annotations, file_id, time):
    frames = []
    frames.append(process_video_frame(annotations, file_id, time-5.0))
    frames.append(process_video_frame(annotations, file_id, time))
    frames.append(process_video_frame(annotations, file_id, time+5.0))
    return frames 


def extract_features_byframe(detector, frame_path):
  df = detector.detect_image(frame_path)
  return df

def extract_features_file(detector, directory, frames):
    dfs = []
    for time, frame in frames:
        file_location = os.path.join(directory, frame)
        df = extract_features_byframe(detector, file_location)
        df['time'] = time
        dfs.append(df)
    
    return dfs

def extract(video_change_points, annotations,detector, split):
    all_features = {}
    for file_id in video_change_points:
        if annotations[file_id]['split'] == split and annotations[file_id]['processed'] == True and (len(video_change_points[file_id]) >0):
            features = {}
            directory = annotations[file_id]['processed_dir']
            time_stamps = video_change_points[file_id]
            for t in time_stamps:
                frames = get_frames(annotations, file_id, t)
                if any(x is None for x in frames) == False:
                    features[t] = extract_features_file(detector, directory, frames) #list of dataframes with features for -5, t, +5 
            all_features[file_id] = features
    return all_features 

def non_change_points(start, end, change_points):
    t = random.uniform(start+5, end-5)
    for change_point in change_points:
        while t <= change_point-5 and t>= change_point+5 :
            t = random.uniform(start+5, end-5)
    return [t]

def get_non_change_points(annotations, video_change_points):
    non_change_points_dict = {}
    for file_id in video_change_points:
        print(file_id)
        start = annotations[file_id]['start']
        end = annotations[file_id]['end']
        non_change_points_dict[file_id] = non_change_points(start, end, video_change_points[file_id])
    return non_change_points_dict

def get_similarity(df, features):
    sim = []
    df = df[features]
    for i in range(len(df.index)-1):
        for j in range(i+1,len(df.index)):
            sim.append(np.dot(df.iloc[i], df.iloc[j])/(np.linalg.norm(df.iloc[i])*np.linalg.norm(df.iloc[j])))
    
    if len(sim) == 0:
        return None
    return sum(sim) / len(sim)

def similarity_scores(features_dict, feature, label):
    df = pd.DataFrame(columns=['file_id','time_stamp', 'sim1',
                             'sim2','sim3', 'label'])
    sims = {}
    for file_id in features_dict:
        for t in features_dict[file_id]:
            frames_t = {}
            sim = []
            frames = features_dict[file_id][t]
            for f in frames:
                if len(f.index) >0:
                    if feature == 'all':
                        features = list(set(f.columns) - set(['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight','FaceScore', 'label', 'input', 'frame']))
                    if feature == 'emotion':
                        features = f.emotions.columns
                    if feature == 'musc':
                        features = f.aus.columns
                    if feature == 'head_pose':
                        features = f.poses.columns
                    sim.append(get_similarity(f, features))

            if any(x is None for x in sim) == False:
                frames_t[t] = sim
                sims[file_id] = frames_t
                row = [file_id, t]
                row.extend(sim)
                row.append(label)
                df.loc[len(df)] = row
    return sims,df

#this function was not used in experiments since this extracts the entire val set and does not first filter for dyadic videos
#running this function for the entire val set can take up to 3 days due slow processing 
def extract_val_set(annotations, detector,split):
    # key is file_id val is a dictioanry with: 'changepoints': changepoints list and 'frames': dictionary of processed frames (key is timestamp, and val is df)
    filtered = {} #dyadic only 
    for file_id in annotations:
        if split == annotations[file_id]['split'] and annotations[file_id]['processed'] == True and annotations[file_id]['data_type'] == 'video':
            filtered[file_id] = {}
            start = annotations[file_id]['start']
            end = annotations[file_id]['end']
            utterances = annotations[file_id]['utterances'] #list of dictionaries with vars for an utterance 
            filtered_frames = {}
            filtered[file_id]['changepoints'] = annotations[file_id]['changepoints']
            filtered[file_id]['frames'] = [] #list of dictionaries with key being the start and end time stamp and value being a list of two dfs containing those features 
            directory = annotations[file_id]['processed_dir']
            for utterance in utterances:
                videos = utterance['video_frames'] #randomly choose a frame that spans an utterance 
                if len(videos) > 0:
                    videos = [videos[0],videos[-1]]
                    span = []
                    for (time,frame) in videos:
                        if start <= time and end >= time:
                            frame_path = os.path.join(directory, frame)
                            df = detector.detect_image(frame_path)
                            span.append(df)
                    if len(span) > 0 and all(len(d) == 2 for d in span):
                        filtered_frames[(videos[0][0],videos[1][0])] = span
                        filtered[file_id]['frames'].append(filtered_frames)
                        
    return filtered_frames 

