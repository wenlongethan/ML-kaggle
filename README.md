# ML-kaggle

Here’s a clean English `README.md` you can drop into the repo:

````markdown
# Truck Fuel Consumption Forecast – Student Solution

This repository contains my solution for the BME “Data Science Competition” homework on the **Truck fuel consumption forecast** Kaggle task.

The goal is to predict `fuel_consumption_sum` for each 1 km segment using telematics and environmental signals.

My final Kaggle submission is an **ensemble of LightGBM and CatBoost**, tuned directly on MAE to match the competition metric.

---

## 1. Repository Structure

```text
truck-fuel-consumption-forecast/
├── README.md
├── report/
│   └── report.pdf                 # 1–3 page project report (for the course)
│
├── data/                          # NOT tracked in git
│   ├── public_train.csv
│   └── public_test.csv
│
├── solutionv5.1_v7.py             # LightGBM main model (L1 objective + OOF target encoding)
├── solutionv7.py                  # CatBoost auxiliary model
├── solutionv1.py                  # Blending script (LGBM + CatBoost)
│
├── submissions/                   # Example outputs
│   ├── submission_v5_1_lgbm_mae_18.9639.csv
│   ├── submission_v7_catboost_mae_19.0721.csv
│   └── submission_blend_55LGB_44Cat.csv   # final best submission
└── requirements.txt
````

> **Note:** `data/` is only a local folder and should be added to `.gitignore`.
> The Kaggle files `public_train.csv` and `public_test.csv` must be placed there manually.

---

## 2. Environment

The code was tested with:

* Python 3.12
* `pandas`
* `numpy`
* `scikit-learn`
* `lightgbm`
* `catboost`
* `matplotlib`, `seaborn` (only for plots, not required for submission)

You can install dependencies via:

```bash
pip install -r requirements.txt
```

(or install the packages manually if needed).

---

## 3. Feature Engineering Overview

All models share the same core feature engineering, implemented directly inside the training scripts:

* **Physical features**

  * `kinetic_energy = weight_1 * speed_mean^2`
  * `momentum = weight_1 * speed_mean`
  * `power_demand = engine_percent_load_at_current_speed_mean * engine_speed_mean`

* **Environment interaction**

  * Convert `env_wind_kph` and `env_sailing_value` to numeric
  * `wind_assist = env_wind_kph * env_sailing_value`

* **OOF Target Encoding (solutionv5.1_v7.py)**
  For high-cardinality categorical columns
  (`driver_name_and_id`, `vehicle_type`, `route_id`, `vehicle_motortype`, `deviceuniquecode`),
  I use 5-fold **out-of-fold target encoding**:

  * For each fold, encode validation rows using means computed on the other folds.
  * For the test set, encode using means computed on the full training data.
  * This avoids target leakage while giving the models strong aggregated signals.

Original ID-like string columns are dropped; the models only see the encoded `_te` features plus numeric signals.

---

## 4. Models

### 4.1 LightGBM – `solutionv5.1_v7.py` (main model)

* Objective: `mae`
* Metric: `mae`
* 5-fold cross validation with shuffling
* Key hyperparameters:

  * `learning_rate = 0.05`
  * `num_leaves = 63`
  * `feature_fraction = 0.9`
  * `bagging_fraction = 0.8`, `bagging_freq = 1`
  * `min_data_in_leaf = 40`
  * `lambda_l2 = 1.0`
* Early stopping with `num_boost_round = 5000` and patience 200.
* The script:

  * Reads `data/public_train.csv` and `data/public_test.csv`
  * Builds features + OOF target encoding
  * Trains LGBM in 5 folds
  * Saves the final prediction as
    `submission_v5_1_lgbm_mae_18.9639.csv`

### 4.2 CatBoost – `solutionv7.py` (auxiliary model)

* Uses CatBoostRegressor with MAE objective.
* Treats the original high-cardinality categorical features directly as CatBoost categories.
* 5-fold cross validation, tuned to get a reasonably strong but diverse model.
* The script saves predictions as
  `submission_v7_catboost_mae_19.0721.csv`.

### 4.3 Blending – `solutionv1.py` (final submission)

* Reads the two single-model submissions:

  ```text
  submission_v5_1_lgbm_mae_18.9639.csv
  submission_v7_catboost_mae_19.0721.csv
  ```

* Blends them by a **weighted average on predictions**:

  [
  \hat{y} = 0.55 \cdot \hat{y}*{LGBM} + 0.44 \cdot \hat{y}*{CatBoost}
  ]

* Writes the final file:

  ```text
  submission_blend_55LGB_44Cat.csv
  ```

This blended submission achieved my **best public leaderboard MAE** and is the solution used for the homework.

---

## 5. How to Reproduce the Final Submission

Assuming you are in the `truck-fuel-consumption-forecast` folder and have put the Kaggle CSVs under `data/`:

```bash
# 1. Train LightGBM and generate its submission
python solutionv5.1_v7.py

# 2. Train CatBoost and generate its submission
python solutionv7.py

# 3. Blend the two submissions into the final file
python solutionv1.py
```

The final file `submission_blend_55LGB_44Cat.csv` is ready to be uploaded to Kaggle.

---

## 6. Notes

* All scripts are written to be **self-contained**: each one reads from `data/` and writes outputs to the project root or `submissions/`.
* I focused on **metric-aligned training (MAE)**, **leak-safe OOF target encoding**, and **small-ensemble blending** to reach a competitive leaderboard score within limited time and hardware resources.

```


```
