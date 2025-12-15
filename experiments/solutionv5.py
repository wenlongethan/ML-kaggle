import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error  # 注意：这次我们导入 MAE
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
# 2. 特征工程 (继承 V4 的强力特征)
# ==========================================
print("正在构建特征 (V5 - MAE Special)...")


def get_target_encoding(train_df, test_df, col, target):
    # 计算均值
    mapping = train_df.groupby(col)[target].mean()
    train_new_col = train_df[col].map(mapping)
    test_new_col = test_df[col].map(mapping)
    # 填充缺失
    global_mean = train_df[target].mean()
    train_new_col = train_new_col.fillna(global_mean)
    test_new_col = test_new_col.fillna(global_mean)
    return train_new_col, test_new_col


# --- 物理特征 ---
for df in [train, test]:
    # 动能与动量
    df['kinetic_energy'] = df['weight_1'] * (df['speed_mean'] ** 2)
    df['momentum'] = df['weight_1'] * df['speed_mean']
    # 功率需求
    df['power_demand'] = df['engine_percent_load_at_current_speed_mean'] * df['engine_speed_mean']

    # 修复环境数据并计算交互
    if 'env_wind_kph' in df.columns and 'env_sailing_value' in df.columns:
        df['env_wind_kph'] = pd.to_numeric(df['env_wind_kph'], errors='coerce').fillna(0)
        df['env_sailing_value'] = pd.to_numeric(df['env_sailing_value'], errors='coerce').fillna(0)
        df['wind_assist'] = df['env_wind_kph'] * df['env_sailing_value']

# --- Target Encoding ---
# 既然 ID 代表 1km 切片，那么 deviceuniquecode (卡车ID) 和 route_id (路线) 就是最关键的
te_cols = ['driver_name_and_id', 'vehicle_type', 'route_id', 'vehicle_motortype', 'deviceuniquecode']
existing_te_cols = [c for c in te_cols if c in train.columns]

print(f"🔥 正在对以下列进行 Target Encoding: {existing_te_cols}")
for col in existing_te_cols:
    train[f'{col}_target_mean'], test[f'{col}_target_mean'] = get_target_encoding(train, test, col, target_col)

# ==========================================
# 3. 筛选与清洗
# ==========================================
drop_cols = ['ID', 'Trip_ID_first', 'Trip_ID_last', target_col]
features = [c for c in train.columns if c not in drop_cols]

# 强制指定类别
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
# 4. 训练 (V5 - 切换为 MAE 目标)
# ==========================================
folds = 5
kf = KFold(n_splits=folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
scores = []
feature_importance_df = pd.DataFrame()

print(f"开始训练 LightGBM (Objective='MAE')...")

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    X_train, y_train = train[features].iloc[train_idx], train[target_col].iloc[train_idx]
    X_val, y_val = train[features].iloc[val_idx], train[target_col].iloc[val_idx]

    # !!! 关键修改 !!!
    params = {
        'objective': 'mae',  # 直接优化 MAE (L1 Loss)
        'metric': 'mae',  # 评估指标也是 MAE
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 64,  # 稍微复杂一点
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

    # 计算 MAE 而不是 RMSE
    score = mean_absolute_error(y_val, val_pred)
    scores.append(score)
    print(f"Fold {fold + 1} MAE: {score:.4f}")

    # 记录重要性
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = features
    fold_importance["importance"] = model.feature_importance()
    fold_importance["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)

# ==========================================
# 5. 结果
# ==========================================
mean_mae = np.mean(scores)
print(f"\n========================================")
print(f"🔥 V5版本 (MAE目标) 平均 MAE: {mean_mae:.4f}")
print(f"========================================")

submission = pd.DataFrame({'ID': test['ID'], target_col: test_preds})
sub_filename = f'submission_v5_mae_{mean_mae:.4f}.csv'
submission.to_csv(sub_filename, index=False)
print(f"提交文件已生成: {sub_filename}")

# 绘图
plt.figure(figsize=(10, 8))
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:20].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
plt.title('Top 20 Features (V5 - MAE Optimized)')
plt.tight_layout()
plt.savefig('feature_importance_v5.png')