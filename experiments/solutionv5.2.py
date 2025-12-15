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

# ==========================================
# 2. 防泄露目标编码 (K-Fold Target Encoding) - 核心技术
# ==========================================
print("正在构建特征 (V6 - OOF Target Encoding)...")


def add_oof_target_encoding(train_df, test_df, cols, target, n_splits=5):
    """
    使用 K-Fold 方式生成目标编码，彻底杜绝数据泄露。
    """
    # 1. 先给 Test 集生成编码 (使用全量 Train 的均值，这是合法的)
    for col in cols:
        # 计算全局均值映射
        global_mean = train_df[target].mean()
        mapping = train_df.groupby(col)[target].mean()

        # 映射到 Test
        test_df[f'{col}_target_mean'] = test_df[col].map(mapping).fillna(global_mean)

        # 初始化 Train 的新列
        train_df[f'{col}_target_mean'] = np.nan

    # 2. 给 Train 集生成编码 (使用 Out-of-Fold 方式)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for tr_idx, val_idx in kf.split(train_df):
        # 切分数据
        X_tr, X_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]

        for col in cols:
            # 只用训练部分(X_tr)来计算均值
            means = X_tr.groupby(col)[target].mean()

            # 映射到验证部分(X_val)
            # 注意：如果验证集里有训练集没见过的类别，填全局均值
            train_df.loc[val_idx, f'{col}_target_mean'] = X_val[col].map(means)

    # 3. 填充 Train 中可能产生的 NaN (比如某折里出现了生僻类别)
    for col in cols:
        global_mean = train_df[target].mean()
        train_df[f'{col}_target_mean'] = train_df[f'{col}_target_mean'].fillna(global_mean)

    return train_df, test_df


# --- A. 物理特征 ---
for df in [train, test]:
    df['kinetic_energy'] = df['weight_1'] * (df['speed_mean'] ** 2)
    df['momentum'] = df['weight_1'] * df['speed_mean']
    df['power_demand'] = df['engine_percent_load_at_current_speed_mean'] * df['engine_speed_mean']

    if 'env_wind_kph' in df.columns and 'env_sailing_value' in df.columns:
        df['env_wind_kph'] = pd.to_numeric(df['env_wind_kph'], errors='coerce').fillna(0)
        df['env_sailing_value'] = pd.to_numeric(df['env_sailing_value'], errors='coerce').fillna(0)
        df['wind_assist'] = df['env_wind_kph'] * df['env_sailing_value']

# --- B. 执行防泄露编码 ---
# 这些是高维类别特征，最容易泄露
te_cols = ['driver_name_and_id', 'vehicle_type', 'route_id', 'vehicle_motortype', 'deviceuniquecode']
existing_te_cols = [c for c in te_cols if c in train.columns]

print(f"🔥 正在执行 OOF Target Encoding (防止泄露): {existing_te_cols}")
# 注意：这一步会比较慢，因为要在内部跑一遍 5折交叉
train, test = add_oof_target_encoding(train, test, existing_te_cols, target_col)

# ==========================================
# 3. 筛选特征
# ==========================================
drop_cols = ['ID', 'Trip_ID_first', 'Trip_ID_last', target_col]
features = [c for c in train.columns if c not in drop_cols]

# 强制指定类别 (辅助 LightGBM)
cat_cols = []
for col in features:
    is_id_col = any(x in col.lower() for x in ['type', 'id', 'code', 'name'])
    is_not_te = '_target_mean' not in col
    if (train[col].dtype == 'object' or is_id_col) and is_not_te:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
        cat_cols.append(col)

print(f"最终特征数: {len(features)}")

# ==========================================
# 4. 训练 (V6 - MAE + Robust CV)
# ==========================================
folds = 5
kf = KFold(n_splits=folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
scores = []

print(f"开始训练 LightGBM (V6 - Robust)...")

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    X_train, y_train = train[features].iloc[train_idx], train[target_col].iloc[train_idx]
    X_val, y_val = train[features].iloc[val_idx], train[target_col].iloc[val_idx]

    params = {
        'objective': 'mae',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 64,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
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
print(f"🔥 V6版本 (无泄露) 平均 MAE: {mean_mae:.4f}")
print(f"========================================")

submission = pd.DataFrame({'ID': test['ID'], target_col: test_preds})
sub_filename = f'submission_v6_robust_mae_{mean_mae:.4f}.csv'
submission.to_csv(sub_filename, index=False)
print(f"✅ 最稳健的提交文件已生成: {sub_filename}")