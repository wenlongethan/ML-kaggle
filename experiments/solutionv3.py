import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
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
# 2. 核心修复：数据类型清洗 (Fix Types)
# ==========================================
def clean_numeric_columns(df):
    # 强制转换这些看似是字符串其实是数字的列
    # 只要列名里包含 humidity, wind, temp, speed 等，都尝试转数字
    for col in df.columns:
        if df[col].dtype == 'object':
            # 尝试转数字，遇到无法转换的变成 NaN
            try_numeric = pd.to_numeric(df[col], errors='coerce')
            # 如果转换后大部分都不是 NaN，说明这一列原本就是数字
            if try_numeric.notna().sum() > len(df) * 0.5:
                print(f"🔧 修复数据类型: {col} (Object -> Float)")
                df[col] = try_numeric
    return df


print("正在清洗数据类型...")
train = clean_numeric_columns(train)
test = clean_numeric_columns(test)

# ==========================================
# 3. 特征工程 (V3版)
# ==========================================
print("正在进行特征工程 V3...")


def engineer_features(df):
    # 1. 物理特征
    df['kinetic_energy'] = df['weight_1'] * (df['speed_mean'] ** 2)
    df['momentum'] = df['weight_1'] * df['speed_mean']

    # 2. 交互特征
    df['power_demand'] = df['engine_percent_load_at_current_speed_mean'] * df['engine_speed_mean']

    # 3. 修复后的环境特征交互
    if 'env_wind_kph' in df.columns and 'env_sailing_value' in df.columns:
        df['wind_assist'] = df['env_wind_kph'] * df['env_sailing_value']

    return df


train = engineer_features(train)
test = engineer_features(test)

# ==========================================
# 4. 类别处理
# ==========================================
drop_cols = ['ID', 'Trip_ID_first', 'Trip_ID_last', target_col]
features = [c for c in train.columns if c not in drop_cols]

# 重新定义类别列，这次不会包含 humidity 了
cat_cols = []
for col in features:
    # 只有真正的 ID 和 文本 才是类别
    if train[col].dtype == 'object' or 'id' in col.lower() or 'code' in col.lower():
        # 再次确认这一列不是 float
        if not pd.api.types.is_float_dtype(train[col]):
            train[col] = train[col].astype('category')
            test[col] = test[col].astype('category')
            cat_cols.append(col)

print(f"使用的特征数量: {len(features)}")
print(f"🔥 真正的类别特征: {cat_cols}")

# ==========================================
# 5. LightGBM 训练 (回归原始目标)
# ==========================================
folds = 5
kf = KFold(n_splits=folds, shuffle=True, random_state=2025)

# 不再使用 Log 变换，直接预测
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
scores = []
feature_importance_df = pd.DataFrame()

print(f"开始训练 LightGBM (V3 - Fixed Types)...")

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    X_train, y_train = train[features].iloc[train_idx], train[target_col].iloc[train_idx]
    X_val, y_val = train[features].iloc[val_idx], train[target_col].iloc[val_idx]

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 40,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1
    }

    model = lgb.train(
        params,
        lgb.Dataset(X_train, y_train, categorical_feature=cat_cols),
        num_boost_round=5000,
        valid_sets=[lgb.Dataset(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(1000)]
    )

    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(test[features]) / folds

    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    scores.append(rmse)
    print(f"Fold {fold + 1} RMSE: {rmse:.4f}")

    # 记录重要性
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = features
    fold_importance["importance"] = model.feature_importance()
    fold_importance["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)

# ==========================================
# 6. 结果
# ==========================================
mean_rmse = np.mean(scores)
print(f"\n========================================")
print(f"🔥 V3版本 (类型修复) 平均 RMSE: {mean_rmse:.4f}")
print(f"========================================")

submission = pd.DataFrame({'ID': test['ID'], target_col: test_preds})
sub_filename = f'submission_v3_fixed_rmse_{mean_rmse:.4f}.csv'
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
plt.title('Top 20 Features (V3 - Fixed Types)')
plt.tight_layout()
plt.savefig('feature_importance_v3.png')