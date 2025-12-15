
````markdown
# Truck Fuel Consumption Forecast

Course project for the **Data Science Competition** at BME.  
The goal is to predict the **total fuel consumption** of a truck trip (in liters) for each 1-km segment, using sensor, vehicle and environmental data.

The final solution achieves a **public leaderboard MAE ≈ 19.65** on Kaggle, using an ensemble of a LightGBM model and a CatBoost model.

---

## 1. Dataset

Competition: **“Truck fuel consumption forecast”** (Kaggle classroom competition).

Download the three CSV files from Kaggle and place them locally as:

```bash
data/
  public_train.csv
  public_test.csv
  public_sample_submission.csv
````

> Note: The `data/` folder in the repository only contains a placeholder file (`.gitkeeper`).
> Raw CSVs are **not** committed due to size and competition rules.

Target column in `public_train.csv`:

* `fuel_consumption_sum` – total fuel consumption for that 1-km segment.

---

## 2. Repository Structure

```text
.
├── README.md
│
├── data/                       # NOT tracked in git (only .gitkeeper)
│   ├── public_train.csv        # downloaded from Kaggle (local only)
│   ├── public_test.csv
│   └── public_sample_submission.csv
│
├── src/
│   └── features/
│       └── build_features.py   # shared feature engineering utilities (if used)
│
├── model/
│   ├── solutionv5.1_v7.py      # blending script (LightGBM + CatBoost)
│   └── solutionv7.py           # CatBoost single model (V7)
│   # (optional: other model scripts can also be placed here)
│
├── experiments/                # older versions / exploratory work
│   ├── solutionv3.py
│   ├── solutionv5.py
│   ├── solutionv6.py
│   ├── solutionv8.py
│   └── ...                     # etc., not used in final submission
│
└── submissions/
    ├── submission_v5_1_lgbm_mae_18.9639.csv   # LightGBM single model
    ├── submission_v7_catboost_mae_19.0721.csv # CatBoost single model
    └── submission_blend_55LGB_44Cat.csv       # ✅ final Kaggle submission
```

Only the **`model/`**, **`src/`**, **`experiments/`**, **`submissions/`** and a placeholder under **`data/`** are pushed to GitHub.
The teacher can reproduce the results by downloading the Kaggle data into `data/`.

---

## 3. Method Overview

### 3.1 Feature Engineering

Feature engineering is done directly inside the model scripts (and optionally in `src/features/build_features.py`):

* **Physical features**

  * `kinetic_energy = weight_1 * speed_mean^2`
  * `momentum = weight_1 * speed_mean`
  * `power_demand = engine_percent_load_at_current_speed_mean * engine_speed_mean`

* **Environment interaction**

  * Clean `env_wind_kph` and `env_sailing_value` and create
    `wind_assist = env_wind_kph * env_sailing_value`.

* **OOF Target Encoding**

  * For high-cardinality categorical columns
    (`driver_name_and_id`, `vehicle_type`, `route_id`, `vehicle_motortype`, `deviceuniquecode`)
  * 5-fold **out-of-fold** target encoding to avoid target leakage:

    * Each fold’s encoding is computed from the other folds.
    * Test set encoding uses global means from the full train set.
  * Original high-cardinality columns are dropped and only the `*_te` columns are kept.

* **Cleaning**

  * Convert candidate numeric columns from string to numeric (`errors='coerce'`).
  * Drop unused IDs such as `ID`, `Trip_ID_first`, `Trip_ID_last` from the feature set.

### 3.2 Models

1. **LightGBM (V5.1) – MAE Objective with OOF Target Encoding**

   * Implemented in: `model/solutionv5.1_v7.py` (LightGBM part)
   * 5-fold CV, KFold with shuffling.
   * Objective / metric: `mae`
   * Typical parameters (simplified):

     ```python
     params = {
         "objective": "mae",
         "metric": "mae",
         "boosting_type": "gbdt",
         "learning_rate": 0.05,
         "num_leaves": 63,
         "feature_fraction": 0.9,
         "bagging_fraction": 0.8,
         "bagging_freq": 1,
         "min_data_in_leaf": 40,
         "lambda_l2": 1.0,
         "verbose": -1,
         "seed": 42,
     }
     ```
   * Early stopping with up to 5000 boosting rounds.
   * Produces:

     * `submission_v5_1_lgbm_mae_18.9639.csv`
     * CV MAE ≈ 18.96

2. **CatBoost (V7)**

   * Implemented in: `model/solutionv7.py`
   * Handles categorical features natively.
   * Also trained with MAE objective (regression).
   * Produces:

     * `submission_v7_catboost_mae_19.0721.csv`

3. **Weighted Blending (Final Model)**

   * Implemented in: `model/solutionv5.1_v7.py`
   * Read the two single-model submissions and blend their predictions:

     ```python
     # 55% LightGBM V5.1 + 44% CatBoost V7
     df_blend_55 = df_lgb.copy()
     df_blend_55[target_col] = (
         df_lgb[target_col] * 0.55 +
         df_cat[target_col] * 0.45
     )
     df_blend_55.to_csv("submissions/submission_blend_55LGB_44Cat.csv", index=False)
     ```
   * This simple linear ensemble improves the public leaderboard score to
     **MAE ≈ 19.6455**, which is used as the final submission.

---

## 4. How to Reproduce

1. **Clone the repository**

   ```bash
   git clone <your_repo_url>.git
   cd truck-fuel-consumption-forecast
   ```

2. **Prepare environment**

   ```bash
   pip install -r requirements.txt
   ```

   (or use `environment.yml` with conda, if provided)

3. **Download data**

   * From the Kaggle competition page download:

     * `public_train.csv`
     * `public_test.csv`
     * `public_sample_submission.csv`
   * Place them under `data/` as described above.

4. **Train single models & generate submissions (optional)**

   ```bash
   # LightGBM V5.1
   python model/solutionv5.1_v7.py  # will train LGBM and save its submission

   # CatBoost V7
   python model/solutionv7.py       # will train CatBoost and save its submission
   ```

5. **Generate final blended submission**

   * Ensure the two single-model CSV files are present under `submissions/`:

     * `submission_v5_1_lgbm_mae_18.9639.csv`
     * `submission_v7_catboost_mae_19.0721.csv`
   * Run:

     ```bash
     python model/solutionv5.1_v7.py  # blending part will create submission_blend_55LGB_44Cat.csv
     ```

6. **Upload to Kaggle**

   * Submit `submissions/submission_blend_55LGB_44Cat.csv`
     as the final prediction file.

---

## 5. Experiments

The `experiments/` folder contains earlier versions and ablation studies:

* **solutionv3** – RMSE-based baseline.
* **solutionv5 / v5.3** – first MAE objective + target encoding attempts.
* **solutionv6** – multi-seed LightGBM ensemble (did not beat the final blend).
* **solutionv8** – trip-level aggregation features; performance was worse than V5.1.

These scripts are not needed for reproduction, but they document the exploration process and show why the final blend of **LightGBM V5.1 + CatBoost V7** was chosen.

```


```
