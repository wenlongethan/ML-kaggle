import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 准备数据
# ==========================================
base_path = '/Users/liangwenlong/study/bme/3/ml_usage/project/truck-fuel-consumption-forecast'
train_path = os.path.join(base_path, 'public_train.csv')
test_path = os.path.join(base_path, 'public_test.csv')

print("正在读取数据...")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
target_col = 'fuel_consumption_sum'
test_ids = test['ID']

# ==========================================
# 2. 特征工程 (保持 V5.1 的精华)
# ==========================================
print("正在构建特征 (V5.3 - Regularized)...")


def add_oof_target_encoding(train_df, test_df, cols, target, n_splits=5):
    # 1. Test Encoding (用全量 Train)
    for col in cols:
        global_mean = train_df[target].mean()
        mapping = train_df.groupby(col)[target].mean()
        test_df[f'{col}_te'] = test_df[col].map(mapping).fillna(global_mean)
        train_df[f'{col}_te'] = np.nan

    # 2. Train Encoding (OOF)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(train_df):
        X_tr, X_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        for col in cols:
            means = X_tr.groupby(col)[target].mean()
            train_df.loc[val_idx, f'{col}_te'] = X_val[col].map(means)

    # 3. Fill NaNs
    for col in cols:
        global_mean = train_df[target].mean()
        train_df[f'{col}_te'] = train_df[f'{col}_te'].fillna(global_mean)

    return train_df, test_df


# --- 物理特征 ---
for df in [train, test]:
    df['kinetic_energy'] = df['weight_1'] * (df['speed_mean'] ** 2)
    df['momentum'] = df['weight_1'] * df['speed_mean']
    df['power_demand'] = df['engine_percent_load_at_current_speed_mean'] * df['engine_speed_mean']

    if 'env_wind_kph' in df.columns and 'env_sailing_value' in df.columns:
        df['env_wind_kph'] = pd.to_numeric(df['env_wind_kph'], errors='coerce').fillna(0)
        df['env_sailing_value'] = pd.to_numeric(df['env_sailing_value'], errors='coerce').fillna(0)
        df['wind_assist'] = df['env_wind_kph'] * df['env_sailing_value']

# --- OOF Target Encoding ---
te_cols = ['driver_name_and_id', 'vehicle_type', 'route_id', 'vehicle_motortype', 'deviceuniquecode']
existing_te_cols = [c for c in te_cols if c in train.columns]
train, test = add_oof_target_encoding(train, test, existing_te_cols, target_col)

# ==========================================
# 3. 筛选与清洗
# ==========================================
drop_cols = ['ID', 'Trip_ID_first', 'Trip_ID_last', target_col]
# 丢弃原始 ID 列，只用 TE 特征，防止过拟合
drop_cols += existing_te_cols

features = [c for c in train.columns if c not in drop_cols]

# 剩余的类别转 category
cat_cols = []
for col in features:
    if train[col].dtype == 'object':
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
        cat_cols.append(col)

print(f"使用的特征数: {len(features)}")

# ==========================================
# 4. 训练 (V5.3 - 强正则化版)
# ==========================================
folds = 5
kf = KFold(n_splits=folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
scores = []

print(f"开始训练 LightGBM (V5.3 Regularized)...")

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    X_train, y_train = train[features].iloc[train_idx], train[target_col].iloc[train_idx]
    X_val, y_val = train[features].iloc[val_idx], train[target_col].iloc[val_idx]

    # !!! 关键改动 !!!
    params = {
        'objective': 'mae',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 40,  # 从 64 降到 40，降低复杂度
        'feature_fraction': 0.7,  # 从 0.9 降到 0.7，增加随机性
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 1.0,  # [新增] L1 正则化
        'lambda_l2': 1.0,  # [新增] L2 正则化
        'min_child_samples': 50,  # [新增] 避免叶子太小
        'verbose': -1,
        'n_jobs': -1
    }

    model = lgb.train(
        params,
        lgb.Dataset(X_train, y_train, categorical_feature=cat_cols),
        num_boost_round=10000,
        valid_sets=[lgb.Dataset(X_val, y_val)],
        callbacks=[lgb.early_stopping(300), lgb.log_evaluation(1000)]
    )

    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(test[features]) / folds

    score = mean_absolute_error(y_val, val_pred)
    scores.append(score)
    print(f"Fold {fold + 1} MAE: {score:.4f}")

# ==========================================
# 5. 结果
# ==========================================
mean_mae = np.mean(scores)
print(f"\n========================================")
print(f"🔥 V5.3 (正则化版) 平均 MAE: {mean_mae:.4f}")
print(f"========================================")

submission = pd.DataFrame({'ID': test_ids, target_col: test_preds})
sub_filename = f'submission_v5_3_reg_mae_{mean_mae:.4f}.csv'
submission.to_csv(sub_filename, index=False)
print(f"提交文件已生成: {sub_filename}")