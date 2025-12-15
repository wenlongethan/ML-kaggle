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
# 2. 特征工程 V2 (增强版)
# ==========================================
print("正在进行特征工程 V2...")


def engineer_features(df):
    # --- 物理特征 (保留 V1 的精华) ---
    # 动能: 0.5 * m * v^2
    df['kinetic_energy'] = df['weight_1'] * (df['speed_mean'] ** 2)
    # 动量: m * v
    df['momentum'] = df['weight_1'] * df['speed_mean']

    # --- 驾驶行为 ---
    # 刹车与油门交互
    df['braking_intensity'] = df['brake_switch_mean'] * df['speed_mean']
    df['pedal_vs_speed'] = df['accelerator_pedal_position_mean'] * df['engine_speed_mean']

    # --- 环境 ---
    # 顺风/逆风系数 (假设 sailing_value 是正向的)
    if 'env_sailing_value' in df.columns:
        df['wind_assist'] = df['env_wind_kph'] * df['env_sailing_value']

    # --- [新增] 交互特征 ---
    # 负载与坡度 (如果有 road_level)
    if 'road_level_approximation' in df.columns:
        df['load_on_slope'] = df['weight_1'] * df['road_level_approximation']

    return df


train = engineer_features(train)
test = engineer_features(test)

# ==========================================
# 3. 数据清洗与类别处理 (关键修复)
# ==========================================
# 剔除不需要的列
drop_cols = ['ID', 'Trip_ID_first', 'Trip_ID_last', target_col]
features = [c for c in train.columns if c not in drop_cols]

# --- 强制指定类别特征 ---
# 哪怕它们看起来像数字，只要代表ID或类型，就转为 category
potential_cats = ['vehicle_type', 'vehicle_motortype', 'driver_name_and_id',
                  'route_id', 'deviceuniquecode']

cat_cols = []
for col in features:
    # 如果列名包含 id, type, code 或者本身就是 object 类型
    if train[col].dtype == 'object' or any(x in col.lower() for x in ['type', 'id', 'code', 'name']):
        # 确保在训练集和测试集都存在
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
        cat_cols.append(col)

print(f"使用的特征数量: {len(features)}")
print(f"🔥 识别到的类别特征 (将被特殊处理): {cat_cols}")

# ==========================================
# 4. LightGBM 训练 (Log 变换 + 更多轮数)
# ==========================================
folds = 5
kf = KFold(n_splits=folds, shuffle=True, random_state=42)

# !!! 核心技巧：对目标变量取 Log，让分布更正态 !!!
# 预测完成后再用 exp 还原
y_target = np.log1p(train[target_col])

oof_preds_log = np.zeros(len(train))
test_preds_log = np.zeros(len(test))
scores = []
feature_importance_df = pd.DataFrame()

print(f"开始训练 LightGBM (CV={folds}, Max Rounds=10000)...")

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    X_train, y_train = train[features].iloc[train_idx], y_target.iloc[train_idx]
    X_val, y_val = train[features].iloc[val_idx], y_target.iloc[val_idx]

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,  # 稍微降低学习率，通过增加轮数来提升精度
        'num_leaves': 64,  # 增加树的复杂度
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': -1
    }

    model = lgb.train(
        params,
        lgb.Dataset(X_train, y_train, categorical_feature=cat_cols),
        num_boost_round=10000,  # !!! 大幅增加上限 !!!
        valid_sets=[lgb.Dataset(X_val, y_val)],
        callbacks=[lgb.early_stopping(300), lgb.log_evaluation(1000)]
    )

    # 预测并还原 (exp)
    val_pred_log = model.predict(X_val)
    oof_preds_log[val_idx] = val_pred_log
    test_preds_log += model.predict(test[features]) / folds

    # 还原到原始尺度计算 RMSE
    val_pred_original = np.expm1(val_pred_log)
    y_val_original = np.expm1(y_val)

    rmse = np.sqrt(mean_squared_error(y_val_original, val_pred_original))
    scores.append(rmse)
    print(f"Fold {fold + 1} RMSE: {rmse:.4f}")

    # 记录重要性
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = features
    fold_importance["importance"] = model.feature_importance()
    fold_importance["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)

# ==========================================
# 5. 结果与绘图
# ==========================================
mean_rmse = np.mean(scores)
print(f"\n========================================")
print(f"🔥 V2版本 平均 RMSE: {mean_rmse:.4f}")
print(f"========================================")

# 生成提交
final_preds = np.expm1(test_preds_log)  # 记得还原
submission = pd.DataFrame({'ID': test['ID'], target_col: final_preds})
sub_filename = f'submission_v2_log_rmse_{mean_rmse:.4f}.csv'
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
plt.title('Top 20 Features (V2 Model)')
plt.tight_layout()
plt.savefig('feature_importance_v2.png')