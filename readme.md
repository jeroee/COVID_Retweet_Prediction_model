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

## Folder Details
|folder | details |
| --- | --- |
| Dataset | raw dataset tweets (download with raw_dataset.sh) |
|processed_dataset | processed dataset split into features.csv and results.csv |
|trained_models | trained model file (best performing) |
|performance_plots | final perforamnce loss graph for our best performing model |
|custom_dataset.py | converts csv files into custome dataset to be used by pytorch |
|data_processing_onehot.py| processes raw dataset tweers into processed dataset features.csv and results.csv |
|model.py| model architecture definition and model training functions |
|utils.py | common functions that is used throughout the pipeline |
|main.ipynb  | training pipeline which takes in processed data features.csv and results.csv to train model |

<!-- ```
Dataset                      - all raw dataset tweets (download with raw_dataset.sh)
processed_dataset            - processed dataset split into features.csv and results.csv
trained_models               - trained model file (best performing)
performance_plots            - final perforamnce loss graph for our best performing model
custom_dataset.py            - converts csv files into custome dataset to be used by pytorch
data_processing_onehot.py    - processes raw dataset tweers into processed dataset features.csv and results.csv
model.py                     - model architecture definition and model training functions
utils.py                     - common functions that is used throughout the pipeline
main.ipynb                   - training pipeline which takes in processed data features.csv and results.csv to train model
``` -->

## GUI Recreation Steps
If you want to recreate our GUI locally, please follow the steps in the following links:

1. Backend: https://github.com/wilbertaristo/COVID_Retweet_Prediction_Backend
2. Frontend: https://github.com/wilbertaristo/COVID-Retweet-Prediction-GUI/tree/local_recreation
