# Traffic Congestion Prediction

COMP903 project — predicts Chicago traffic congestion using Logistic Regression, Gradient Boosting, and LSTM models, served via a Streamlit web app.

## Models

| Model | Predicts | Features |
|-------|----------|----------|
| Logistic Regression | Current-period congestion | Hour, Day of week |
| Gradient Boosting | Current-period congestion | Hour, Day of week |
| LSTM | Next-period congestion | Speed, Hour, Day of week, Speed delta, Rolling mean (6-step sequence) |

Congestion is defined as speed < 20 mph.

## Setup

```bash
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

The app loads `traffic_sample.csv` from the project root. On first run, LR and GB models are trained and saved to `lr_model.joblib` / `gb_model.joblib`. Subsequent runs load from disk.

## Training the LSTM

From the `lstm/` directory:

```bash
pip install -r requirements.txt
python lstm_model.py
```

The script looks for the full Chicago dataset CSV in the project root, falling back to `traffic_sample.csv`. The trained model is saved as `lstm_model.h5` in the project root.

## Dataset

- `traffic_sample.csv` — 50,000-row sample used by the app
- Full Chicago Traffic Tracker dataset (190 MB, not tracked in git) — used for LSTM training

## Project structure

```
traffic_prediction/
├── app.py                  # Streamlit web app
├── main_copy.py            # Standalone LR exploration script
├── lstm/
│   ├── lstm_model.py       # LSTM training script
│   └── requirements.txt
├── lstm_model.h5           # Trained LSTM model
├── traffic_sample.csv      # Sample dataset
└── requirements.txt
```
