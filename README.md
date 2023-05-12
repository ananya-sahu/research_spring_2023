# research_spring_2023
this is the readme for visual entrainment and acoustic entrainment research

Library Installation requirements:
Install Py-feat: https://py-feat.org/pages/installation.html
pip install py-feat
Install Sklearn: https://scikit-learn.org/stable/install.html
pip install -U scikit-learn

To run feature extraction from frames: 
(replace file names with your specified paths to generate pkl and csv files) 
python3 example_run.py
This will generate four main files that will be needed for model training and inference 
- for train data: change_point_features_train.pkl, non_change_point_features_train.pkl, train_time_scalars.pkl
- for val data: cp_features_val_dyadic.pkl, ncp_val.pkl, is_val.pkl

To run model training and predictions
- input the path to the generated pickle files (specified in notebook which files go in which sections) 
- run cell by cell 
- for model predictions output files will be generated to get the AP scores 

Get AP scores: 
- create a folder called apscoring and move the generated outputs from model predictions to the folder 
- make another folder called preds_refs within the apscoring folder 
- run python3 convert_preds.py, this will generate predicition dictionaries using the outputs and reference dictionary
- then run python3 score.py which will print the name of the file it is scoring as well as the score 

Extra: 
filter.py: run filter.py after generating the changepoint features to ensure only dyadic videos are being processed for the final train and val sets
analysis.ipynb: contains code for visualizing similarity changes in changepoint vs nonchangepoint annotations as well as prelimnary models that use just similarity scores to predict changepoints in the subset of the train data 
gen_plots.ipynb: contains code to visualize similarity changes in changepoint predictions by impact scalar 
facial_feature_extraction_pipeline (1).ipynb: contains code for an segmenting a video and processing the entire video rather than frames with Py-feat

