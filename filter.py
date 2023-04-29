import pickle 

with open("/home/as5957/research_spring_2023/change_point_features_test.pkl", 'rb') as f:
    all_features = pickle.load(f)

print(len(all_features))

filtered = {}
for id in all_features:
    change_points = all_features[id]
    filtered_time = {}
    for time in change_points:
        dfs = change_points[time]
        if all(len(d) == 2 for d in dfs):
            filtered_time[time] = dfs
    if len(filtered_time) >0:
        filtered[id] = filtered_time

print(len(filtered))

with open("./change_points_test_two.pkl", 'wb') as f:
    pickle.dump(filtered, f)