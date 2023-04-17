import pandas as pd
import numpy as np
import os 
import pickle 
import random
from collections import defaultdict
from feat import Detector 

 
#filter annotations for only video type data 
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
    
    # df_all = pd.concat(dfs)
    # return df_all
    return dfs


def get_similarity(df, features):
    sim = []
    df = df[features]
    for i in range(len(df.index)-1):
        for j in range(i+1,len(df.index)):
            sim.append(np.dot(df.iloc[i], df.iloc[j])/(np.linalg.norm(df.iloc[i])*np.linalg.norm(df.iloc[j])))
    
    return sum(sim) / len(sim)

def extract(video_change_points, annotations,detector, split):
    all_features = {}
    for file_id in annotations:
        print(file_id)
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
            # random.randint(start+5, end-5)
    return t

def get_non_change_points(annotations, video_change_points):
    non_change_points_dict = {}
    for file_id in video_change_points:
        print(file_id)
        start = annotations[file_id]['start']
        end = annotations[file_id]['end']
        non_change_points_dict[file_id] = non_change_points(start, end, video_change_points[file_id])
    return non_change_points

def main():
    #load original annotations
    # with open('/mnt/swordfish-pool2/ccu/as5957-cache.pkl', 'rb') as handle:
    #     annotations = pickle.load(handle)
    
    #filter and save video only annotations
    # annotations = filter_only_video(annotations)
    # with open("./video_annotations.pkl", 'wb') as f:
    #         pickle.dump(annotations, f)
    
    with open('/home/as5957/research_spring_2023/video_annotations.pkl', 'rb') as handle:
        annotations = pickle.load(handle)
    
    #get changepoints a video file and add as a list 
    # change_points = defaultdict(list)
    # for file_id in annotations:
    #     changes = get_change_points(annotations,file_id)
    #     for c in changes:
    #         change_points[file_id].append(c['timestamp'])

    # with open("./video_changepoints.pkl", 'wb') as f:
    #         pickle.dump(change_points, f)

   #{'file_id': [timestamps of change points]}
    with open("./video_changepoints.pkl", 'rb') as handle:
        video_change_points = pickle.load(handle)
    
    # frames = get_frames(annotations, 'M01000AJ9', 94.0)
    # directory = annotations['M01000AJ9']['processed_dir']
    
    # features = extract_features_file(detector, directory, frames)
    # features.to_csv("extracted_test.csv")

    #get change_point features
    # detector = Detector()
    # change_point_features = extract(video_change_points,annotations,detector,'train')
    # with open("./change_point_features.pkl", 'wb') as f:
    #     pickle.dump(change_point_features, f)

    #get non change points 
    # non_change_points = get_non_change_points(annotations, video_change_points)
    # with open("./non_change_points.pkl", 'wb') as f:
    #     pickle.dump(non_change_points, f)
    
    #get non change_point features
    with open("./non_change_points.pkl", 'rb') as handle:
        non_change_points = pickle.load(handle)
    
    detector = Detector()
    non_change_point_features = extract(non_change_points,annotations,detector,'train')
    with open("./non_change_point_features.pkl", 'wb') as f:
        pickle.dump(non_change_point_features, f)


    
    





if __name__ == "__main__":
    main()

