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

# 目标变量
target_col = 'fuel_consumption_sum'

# ==========================================
# 2. 特征工程 (Feature Engineering) - 提分核心
# ==========================================
print("正在进行特征工程...")


def engineer_features(df):
    # --- 物理特征 ---
    # 动能代理变量 (Kinetic Energy Proxy): 0.5 * m * v^2
    # 假设 weight_1 是主要重量
    df['kinetic_energy'] = df['weight_1'] * (df['speed_mean'] ** 2)

    # 动量代理变量 (Momentum Proxy): m * v
    df['momentum'] = df['weight_1'] * df['speed_mean']

    # --- 驾驶行为特征 ---
    # 刹车强度: 速度越高，刹车越浪费能量
    df['braking_intensity'] = df['brake_switch_mean'] * df['speed_mean']

    # 巡航效率: 巡航开启且速度较高时最省油
    df['cruise_efficiency'] = df['cruise_control_active_mean'] * df['speed_mean']

    # --- 发动机特征 ---
    # 功率需求代理: 负载 * 转速
    df['power_demand'] = df['engine_percent_load_at_current_speed_mean'] * df['engine_speed_mean']

    # --- 环境特征 ---
    # 风的影响: 风速 * 顺风系数 (假设 sailing_value 越大越顺风)
    # 如果 sailing_value 是这一列的名字 'env_sailing_value'
    if 'env_sailing_value' in df.columns:
        df['wind_impact'] = df['env_wind_kph'] * df['env_sailing_value']

    return df


train = engineer_features(train)
test = engineer_features(test)

# ==========================================
# 3. 数据清洗与编码
# ==========================================
# 剔除不需要的列 (ID类, 目标列)
drop_cols = ['ID', 'Trip_ID_first', 'Trip_ID_last', target_col]
# 注意：测试集没有 target_col，所以只drop ID类
features = [c for c in train.columns if c not in drop_cols]

# 找出类别列 (Categorical Columns)
cat_cols = []
for col in features:
    # 如果是字符串对象，或者列名里包含 ID/name 但不是主要ID
    if train[col].dtype == 'object' or 'id' in col.lower() or 'code' in col.lower():
        # 排除掉数值型的 ID 误判，这里主要处理 driver_name_and_id, vehicle_type 等
        if train[col].dtype == 'object':
            train[col] = train[col].astype('category')
            test[col] = test[col].astype('category')
            cat_cols.append(col)

print(f"使用的特征数量: {len(features)}")
print(f"类别特征: {cat_cols}")

# ==========================================
# 4. LightGBM 模型训练 (5折交叉验证)
# ==========================================
folds = 5
kf = KFold(n_splits=folds, shuffle=True, random_state=2025)

# 存储结果
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
scores = []
feature_importance_df = pd.DataFrame()

print(f"开始训练 LightGBM (CV={folds})...")

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    X_train, y_train = train[features].iloc[train_idx], train[target_col].iloc[train_idx]
    X_val, y_val = train[features].iloc[val_idx], train[target_col].iloc[val_idx]

    # LightGBM 参数 (针对回归优化)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,  # 较小的学习率更稳
        'num_leaves': 40,  # 稍微增加复杂度
        'feature_fraction': 0.8,  # 每次随机选80%特征
        'bagging_fraction': 0.8,  # 每次随机选80%数据
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1
    }

    model = lgb.train(
        params,
        lgb.Dataset(X_train, y_train, categorical_feature=cat_cols),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]  # 不刷屏，只显示结果
    )

    # 预测
    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(test[features]) / folds

    # 记录分数
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    scores.append(rmse)
    print(f"Fold {fold + 1} RMSE: {rmse:.4f}")

    # 记录特征重要性 (为了写报告)
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = features
    fold_importance["importance"] = model.feature_importance()
    fold_importance["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)

# ==========================================
# 5. 结果分析与提交
# ==========================================
mean_rmse = np.mean(scores)
print(f"\n========================================")
print(f"🔥 本地验证集平均 RMSE: {mean_rmse:.4f}")
print(f"========================================")

# 保存提交文件
submission = pd.DataFrame({
    'ID': test['ID'],
    target_col: test_preds
})
# 生成一个带分数的文件名，方便你区分版本
sub_filename = f'submission_lgbm_rmse_{mean_rmse:.4f}.csv'
submission.to_csv(sub_filename, index=False)
print(f"✅ 提交文件已生成: {sub_filename}")

# ==========================================
# 6. 生成特征重要性图表 (写报告神器)
# ==========================================
plt.figure(figsize=(10, 8))
# 取平均重要性
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:20].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
plt.title('Top 20 Most Important Features for Fuel Consumption')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("✅ 特征重要性图表已保存: feature_importance.png")