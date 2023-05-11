import pandas as pd 
import os
import pickle


def preds_refs(file_name):
    print(file_name)
    df = pd.read_csv(file_name)
    df1 = df[df['true'] == 1]
    refs = df1[['time','file_id','impact_scalar_num','type']]
    preds = df[['time','file_id','type','llr']]
    
    refs = refs.rename(columns={"time": "timestamp", "impact_scalar_num": "impact_scalar"})

    preds = preds.rename(columns={"time": "timestamp", "llhood": "llr"})

    refs = refs.to_dict('records')
    preds = preds.to_dict('records')

    return refs,preds
    

def main():
    directory = '/home/as5957/research_spring_2023/apscoring'
    path = '/home/as5957/research_spring_2023/apscoring/preds_refs'
    refs_save = None
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            refs,preds = preds_refs(f)
            new_file = filename.replace('.csv', '.pkl')
            save_preds = os.path.join(path, new_file)
            print(save_preds)
            with open(save_preds, 'wb') as handle:
                pickle.dump(preds, handle, protocol=pickle.HIGHEST_PROTOCOL)
            refs_save = refs
   
    with open('/home/as5957/research_spring_2023/apscoring/preds_refs/refs.pkl', 'wb') as handle:
            pickle.dump(refs_save, handle, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == "__main__":
    main()

