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
    
    # df_all = pd.concat(dfs)
    # return df_all
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
            # random.randint(start+5, end-5)
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
                        # features = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise','neutral']
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

def main():
    #load original annotations
    with open('/mnt/swordfish-pool2/ccu/as5957-cache.pkl', 'rb') as handle:
        annotations = pickle.load(handle)
    
    #filter and save video only annotations
    # annotations = filter_only_video(annotations)
    # with open("./video_annotations.pkl", 'wb') as f:
    #         pickle.dump(annotations, f)
    
    # with open('/home/as5957/research_spring_2023/video_annotations.pkl', 'rb') as handle:
    #     annotations = pickle.load(handle)
    
    #get changepoints a video file and add as a list 
    # change_points = defaultdict(list)
    # for file_id in annotations:
    #     changes = get_change_points(annotations,file_id)
    #     for c in changes:
    #         change_points[file_id].append(c['timestamp'])

    # with open("./video_changepoints.pkl", 'wb') as f:
    #         pickle.dump(change_points, f)

   #{'file_id': [timestamps of change points]}
    # with open("./video_changepoints.pkl", 'rb') as handle:
    #     video_change_points = pickle.load(handle)
    
    # subset = {k: video_change_points[k] for k in list(video_change_points)[:5]}

    
    # frames = get_frames(annotations, 'M01000AJ9', 94.0)
    # directory = annotations['M01000AJ9']['processed_dir']
    
    # features = extract_features_file(detector, directory, frames)
    # features.to_csv("extracted_test.csv")

    #get change_point features
    detector = Detector()
    # change_point_features_val = extract(video_change_points,annotations,detector,'test')
    # with open("./change_point_features_test.pkl", 'wb') as f:
    #     pickle.dump(change_point_features_val, f)
    dyadic_val_set = extract_val_set(annotations, detector,'val')
    with open("./val_set.pkl", 'wb') as f:
        pickle.dump(dyadic_val_set, f)
    
   


    # get non change points 
    # non_change_points_dict = get_non_change_points(annotations, video_change_points)
    # print(non_change_points_dict)
    # with open("./non_change_points.pkl", 'wb') as f:
    #     pickle.dump(non_change_points_dict, f)
    
    #get non change_point features
    # with open("./non_change_points.pkl", 'rb') as handle:
    #     non_change_points_dict = pickle.load(handle)
    
    # detector = Detector()
    # non_change_point_features_val = extract(non_change_points_dict,annotations,detector,'test')
    # with open("./non_change_point_features_test.pkl", 'wb') as f:
    #     pickle.dump(non_change_point_features_val, f)
    

    #load frames and get similarities
    # with open("./non_change_point_features.pkl", 'rb') as handle:
    #     non_change_points_frames = pickle.load(handle)

    # with open("./change_point_features.pkl", 'rb') as handle:
    #     change_points_frames = pickle.load(handle)
    
    # sims_dict_cp, sims_df_cp = similarity_scores(change_points_frames,'head_pose', 'cp')
    # sims_dict_ncp, sims_df_ncp = similarity_scores(non_change_points_frames,'head_pose', 'ncp')
    
    # with open("./non_change_point_sims_pose.pkl", 'wb') as f:
    #     pickle.dump(sims_dict_ncp, f)

    # with open("./change_point_sims_pose.pkl", 'wb') as f:
    #     pickle.dump(sims_dict_cp, f)
    

    # sims_df_cp.to_csv("./change_point_sims_pose.csv")
    # sims_df_ncp.to_csv("./nonchange_point_sims_pose.csv")


    # change_point_all_sims = pd.read_csv('/home/as5957/research_spring_2023/change_point_sims.csv', index_col=0)
    # change_point_emotion_sims = pd.read_csv('/home/as5957/research_spring_2023/change_point_sims_emotions.csv', index_col=0)
    # change_point_musc_sims = pd.read_csv('/home/as5957/research_spring_2023/change_point_sims_musc.csv', index_col=0)
    # change_point_pose_sims = pd.read_csv('/home/as5957/research_spring_2023/change_point_sims_pose.csv', index_col=0)

    # time_stamp_scalars = get_impact_scalars(annotations, list(change_point_all_sims['file_id']))
    # print(time_stamp_scalars.keys())

    # change_point_all_sims = append_impact_scalars(change_point_all_sims, time_stamp_scalars)
    # change_point_all_sims.to_csv("./change_point_all_sims_scalars.csv",index=False)

    # change_point_emotion_sims = append_impact_scalars(change_point_emotion_sims, time_stamp_scalars)
    # change_point_emotion_sims.to_csv("./change_point_all_emotions_scalars.csv",index=False)
    
    # change_point_musc_sims = append_impact_scalars(change_point_musc_sims, time_stamp_scalars)
    # change_point_musc_sims.to_csv("./change_point_musc_sims_scalars.csv",index=False)

    # change_point_pose_sims = append_impact_scalars(change_point_pose_sims, time_stamp_scalars)
    # change_point_pose_sims.to_csv("./change_point_pose_sims_scalars.csv",index=False)


    # with open('/home/as5957/research_spring_2023/non_dyadic_data/change_points_train_non_dyadic.pkl', 'rb') as handle:
    #     train = pickle.load(handle)
    
    # with open('/home/as5957/research_spring_2023/non_dyadic_data/change_points_val_non_dyadic.pkl', 'rb') as handle:
    #     val = pickle.load(handle)
    
    # with open('/home/as5957/research_spring_2023/non_dyadic_data/change_points_test_non_dyadic.pkl', 'rb') as handle:
    #     test = pickle.load(handle)
    
    # time_stamp_scalars_train = get_impact_scalars(annotations, list(train.keys()))
    # time_stamp_scalars_val = get_impact_scalars(annotations, list(val.keys()))
    # time_stamp_scalars_test = get_impact_scalars(annotations, list(test.keys()))

    # with open("/home/as5957/research_spring_2023/train_nd.pkl", 'wb') as f:
    #     pickle.dump(time_stamp_scalars_train, f)
    
    # with open("/home/as5957/research_spring_2023/val_nd.pkl", 'wb') as f:
    #     pickle.dump(time_stamp_scalars_val, f)
    
    # with open("/home/as5957/research_spring_2023/test_nd.pkl", 'wb') as f:
    #     pickle.dump(time_stamp_scalars_test, f)

if __name__ == "__main__":
    main()

