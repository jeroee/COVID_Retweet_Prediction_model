# COVID19 Retweet Prediction
Predictive model to predict number of retweets based on tweet.
Current code in our files (main.ipynb and model.py) are implemented for the setup for our best performing model.
Hyperparameters and features can be changed according to the report to replicate the experimented models.

## Installation
To download the raw tweet dataset
```bash
cd Dataset
bash raw_dataset.sh
```

## Instructions
1. Dataset                      - All raw dataset tweets
2. processed_dataset            - Processed dataset split into features.csv and results.csv
3. trained_models               -  trained model file (best performing)
4. custom_dataset.py            - converts csv files into custome dataset to be used by pytorch
5. data_processing_onehot.py    - Processes raw dataset tweers into processed dataset features.csv and results.csv
6. model.py                     - model architecture definition and model training functions
7. utils.py                     - common functions that is used throughout the pipeline
9. main.ipynb                   - pipeline to take in processed data features.csv and results.csv to train model

## GUI Recreation Steps
If you want to recreate our GUI locally, please follow the steps in the following links:

1. Backend: https://github.com/wilbertaristo/COVID_Retweet_Prediction_Backend
2. Frontend: https://github.com/wilbertaristo/COVID-Retweet-Prediction-GUI/tree/local_recreation
