import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
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

# 保存 ID 用于提交
test_ids = test['ID']

# ==========================================
# 2. 特征工程 (物理特征 + 简单清洗)
# ==========================================
print("正在构建特征 (V7 - CatBoost)...")


def engineer_features(df):
    # --- 物理特征 ---
    # 动能: 0.5 * m * v^2
    df['kinetic_energy'] = df['weight_1'] * (df['speed_mean'] ** 2)
    # 动量: m * v
    df['momentum'] = df['weight_1'] * df['speed_mean']
    # 功率需求
    df['power_demand'] = df['engine_percent_load_at_current_speed_mean'] * df['engine_speed_mean']

    # 环境交互
    if 'env_wind_kph' in df.columns and 'env_sailing_value' in df.columns:
        # 简单清洗一下，CatBoost 甚至可以容忍 NaN，但转数字更好
        df['env_wind_kph'] = pd.to_numeric(df['env_wind_kph'], errors='coerce').fillna(0)
        df['env_sailing_value'] = pd.to_numeric(df['env_sailing_value'], errors='coerce').fillna(0)
        df['wind_assist'] = df['env_wind_kph'] * df['env_sailing_value']

    return df


train = engineer_features(train)
test = engineer_features(test)

# ==========================================
# 3. 智能类别处理 (CatBoost 的核心)
# ==========================================
# 剔除无关列
drop_cols = ['ID', 'Trip_ID_first', 'Trip_ID_last', target_col]
features = [c for c in train.columns if c not in drop_cols]

# 找出所有的类别列 (字符串)
cat_features_indices = []
for i, col in enumerate(features):
    # 只要是 object 类型，CatBoost 就能自动处理
    if train[col].dtype == 'object':
        # 填充缺失值为 "Missing"，CatBoost 喜欢字符串
        train[col] = train[col].fillna("Missing").astype(str)
        test[col] = test[col].fillna("Missing").astype(str)
        cat_features_indices.append(col)

print(f"🔥 CatBoost 将自动处理以下类别特征: {cat_features_indices}")

# ==========================================
# 4. 训练 CatBoost (目标: MAE)
# ==========================================
folds = 5
kf = KFold(n_splits=folds, shuffle=True, random_state=2025)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
scores = []

print(f"开始训练 CatBoost (V7)...")

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    X_train, y_train = train[features].iloc[train_idx], train[target_col].iloc[train_idx]
    X_val, y_val = train[features].iloc[val_idx], train[target_col].iloc[val_idx]

    # CatBoost 专用数据池
    train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_features_indices)

    model = CatBoostRegressor(
        iterations=3000,  # 训练轮数
        learning_rate=0.05,  # 学习率
        depth=8,  # 树深一点，捕捉复杂关系
        loss_function='MAE',  # 直接优化 MAE
        eval_metric='MAE',
        random_seed=42,
        verbose=500,  # 每500轮打印一次
        early_stopping_rounds=200,
        task_type="CPU"  # M4 芯片跑 CPU 非常快
    )

    model.fit(train_pool, eval_set=val_pool)

    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(test[features]) / folds

    score = mean_absolute_error(y_val, val_pred)
    scores.append(score)
    print(f"Fold {fold + 1} MAE: {score:.4f}")

# ==========================================
# 5. 结果与提交
# ==========================================
mean_mae = np.mean(scores)
print(f"\n========================================")
print(f"🔥 V7版本 (CatBoost) 平均 MAE: {mean_mae:.4f}")
print(f"========================================")

submission = pd.DataFrame({'ID': test_ids, target_col: test_preds})
sub_filename = f'submission_v7_catboost_mae_{mean_mae:.4f}.csv'
submission.to_csv(sub_filename, index=False)
print(f"✅ 提交文件已生成: {sub_filename}")