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
To download the processed dataset (running data_processing_onehot.py takes 10hrs lol)
```bash
git lfs install
git lfs checkout
```



## GUI Recreation Steps
If you want to recreate our GUI locally, please follow the steps in the following links:

1. Backend: https://github.com/wilbertaristo/COVID_Retweet_Prediction_Backend
2. Frontend: https://github.com/wilbertaristo/COVID-Retweet-Prediction-GUI/tree/local_recreation
