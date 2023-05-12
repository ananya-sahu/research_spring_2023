import os
import pickle
from scoring.average_precision import calculate_average_precision

with open('/home/as5957/research_spring_2023/apscoring/preds_refs/refs.pkl', 'rb') as file:
        refs = pickle.load(file)

directory = '/home/as5957/research_spring_2023/apscoring/preds_refs'
for filename in os.listdir(directory):
    if 'ref' not in filename:
        f = os.path.join(directory, filename)
        print(f)
        if os.path.isfile(f):
            with open(f, 'rb') as file:
                preds = pickle.load(file)
            APs = calculate_average_precision(
        refs, preds,
        text_char_threshold=100,
        time_sec_threshold=10,
        filtering='none',
        n_jobs=1)

    
        print(APs)

