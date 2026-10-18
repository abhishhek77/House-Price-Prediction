# House Price Prediction

Modular scikit-learn project.

```
house_price_prediction/
├── data/house_data.csv      # your dataset (auto-generated sample if missing)
├── models/                  # saved model
├── src/
│   ├── config.py            # paths, column names, settings
│   ├── data.py              # data loading / sample data
│   ├── preprocess.py        # cleaning, split, feature pipeline
│   ├── model.py             # candidate models + selection
│   ├── evaluate.py          # MAE / RMSE / R²
│   └── predict.py           # load model & predict
├── train.py                 # run training
└── app.py                   # example prediction
```

## Run
```
pip install -r requirements.txt
python train.py
python app.py
```

## Use your own data
Put a CSV at `data/house_data.csv`, then edit `NUMERIC_FEATURES`,
`CATEGORICAL_FEATURES` and `TARGET` in `src/config.py` to match its columns.
