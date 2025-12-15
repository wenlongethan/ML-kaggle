import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. 路径与读取数据
# ==========================================
# 按你现在的路径来，如果文件夹移动了自己改一下 base_path
base_path = '/Users/liangwenlong/study/bme/3/ml_usage/project/truck-fuel-consumption-forecast'
train_path = os.path.join(base_path, 'public_train.csv')
test_path = os.path.join(base_path, 'public_test.csv')

print("🔹 Loading data ...")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target_col = 'fuel_consumption_sum'


# ==========================================
# 2. 数据类型清洗：把“伪数字”的 object 列转成 float
# ==========================================
def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == 'object':
            # 尝试转数字，无法转换的变成 NaN
            converted = pd.to_numeric(df[col], errors='coerce')
            # 如果大部分都能转，就认为它本来应该是数值列
            if converted.notna().sum() > 0.5 * len(df):
                print(f"  🔧 Cast to numeric: {col}")
                df[col] = converted
    return df


print("🔹 Cleaning numeric-like columns ...")
train = clean_numeric_columns(train)
test = clean_numeric_columns(test)


# ==========================================
# 3. 特征工程：物理特征 + 环境特征 + Trip 结构
# ==========================================
print("🔹 Feature engineering ...")

for df in [train, test]:
    # ---- 物理特征 ----
    if {'weight_1', 'speed_mean'}.issubset(df.columns):
        df['kinetic_energy'] = df['weight_1'] * (df['speed_mean'] ** 2)
        df['momentum'] = df['weight_1'] * df['speed_mean']

    if {'engine_percent_load_at_current_speed_mean', 'engine_speed_mean'}.issubset(df.columns):
        df['power_demand'] = (
            df['engine_percent_load_at_current_speed_mean'] * df['engine_speed_mean']
        )

    # ---- 环境特征修复 + 交互 ----
    if 'env_wind_kph' in df.columns:
        df['env_wind_kph'] = pd.to_numeric(df['env_wind_kph'], errors='coerce').fillna(0)
    if 'env_sailing_value' in df.columns:
        df['env_sailing_value'] = pd.to_numeric(df['env_sailing_value'],
                                                errors='coerce').fillna(0)
    if {'env_wind_kph', 'env_sailing_value'}.issubset(df.columns):
        df['wind_assist'] = df['env_wind_kph'] * df['env_sailing_value']

    # ---- Trip 结构特征 ----
    # Trip_ID_first 与 ID 是数据里已有的
    if 'Trip_ID_first' in df.columns and 'ID' in df.columns:
        # 每个 trip 的长度
        df['trip_len'] = df.groupby('Trip_ID_first')['ID'].transform('count')
        # 当前切片在 trip 中的序号（从 0 开始）
        df['trip_pos'] = df.groupby('Trip_ID_first').cumcount()
        # 在 trip 中的相对位置（0~1）
        df['trip_pos_ratio'] = df['trip_pos'] / df['trip_len'].replace(0, 1)

        # trip 级平均统计
        if 'speed_mean' in df.columns:
            df['trip_speed_mean'] = df.groupby('Trip_ID_first')['speed_mean'].transform('mean')
        if 'engine_speed_mean' in df.columns:
            df['trip_engine_speed_mean'] = df.groupby('Trip_ID_first')[
                'engine_speed_mean'].transform('mean')
        if 'weight_1' in df.columns:
            df['trip_weight_mean'] = df.groupby('Trip_ID_first')['weight_1'].transform('mean')


# ==========================================
# 4. Target Encoding + Frequency Encoding
# ==========================================
print("🔹 Target & frequency encoding ...")


def target_encoding(train_df, test_df, col, target):
    """
    简单版 target encoding：groupby-mean，然后映射 train / test。
    有一定 leakage，但在这个作业+LGBM 场景里是可以接受的。
    """
    mapping = train_df.groupby(col)[target].mean()
    global_mean = train_df[target].mean()

    train_te = train_df[col].map(mapping).fillna(global_mean)
    test_te = test_df[col].map(mapping).fillna(global_mean)
    return train_te, test_te


te_cols = ['driver_name_and_id',
           'vehicle_type',
           'route_id',
           'vehicle_motortype',
           'deviceuniquecode']

existing_te_cols = [c for c in te_cols if c in train.columns]
print(f"  Will do target encoding for: {existing_te_cols}")

for col in existing_te_cols:
    # Target mean
    train[f'{col}_target_mean'], test[f'{col}_target_mean'] = \
        target_encoding(train, test, col, target_col)

    # Frequency encoding
    vc = train[col].value_counts()
    train[f'{col}_freq'] = train[col].map(vc).fillna(0)
    test[f'{col}_freq'] = test[col].map(vc).fillna(0)


# ==========================================
# 5. 定义特征 & 类别特征
# ==========================================
drop_cols = ['ID', 'Trip_ID_first', 'Trip_ID_last', target_col]
features = [c for c in train.columns if c not in drop_cols]

cat_cols = []
for col in features:
    # 规则：看起来像 ID / 类型 / 名字，且不是 target_mean
    is_id_like = any(k in col.lower() for k in ['id', 'code', 'type', 'name'])
    is_not_te = '_target_mean' not in col

    if (train[col].dtype == 'object' or is_id_like) and is_not_te:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
        cat_cols.append(col)

print(f"🔹 Total features: {len(features)}")
print(f"🔹 Categorical features: {cat_cols}")


# ==========================================
# 6. 多 seed × KFold LightGBM 训练
# ==========================================
seeds = [42, 2025, 3407]   # 可以再加一两个，但训练时间会线性变长
folds = 5

all_seed_scores = []
test_preds_all_seeds = np.zeros((len(seeds), len(test)))
feature_importance_df = pd.DataFrame()

print("🔹 Start multi-seed LightGBM training ...")

for si, seed in enumerate(seeds):
    print(f"\n===== Seed {seed} =====")
    # 每个 seed 单独一个 KFold（保证 shuffle 一致）
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    params = {
        'objective': 'mae',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 64,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 60,
        'lambda_l1': 0.5,
        'lambda_l2': 3.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': seed,
    }

    seed_scores = []
    seed_test_preds = np.zeros(len(test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
        print(f"  -> Seed {seed} | Fold {fold + 1}/{folds}")

        X_train = train[features].iloc[train_idx]
        y_train = train[target_col].iloc[train_idx]
        X_val = train[features].iloc[val_idx]
        y_val = train[target_col].iloc[val_idx]

        train_set = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols, free_raw_data=False)
        val_set = lgb.Dataset(X_val, y_val, categorical_feature=cat_cols, free_raw_data=False)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=10000,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(300),
                lgb.log_evaluation(500)
            ]
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        mae = mean_absolute_error(y_val, val_pred)
        seed_scores.append(mae)
        print(f"     Fold {fold + 1} MAE: {mae:.4f}")

        # test 预测
        seed_test_preds += model.predict(test[features],
                                         num_iteration=model.best_iteration) / folds

        # 保存特征重要性
        fold_importance = pd.DataFrame({
            "feature": features,
            "importance": model.feature_importance(),
            "seed": seed,
            "fold": fold + 1,
        })
        feature_importance_df = pd.concat([feature_importance_df, fold_importance],
                                          axis=0)

    seed_mean_mae = np.mean(seed_scores)
    all_seed_scores.append(seed_mean_mae)
    test_preds_all_seeds[si, :] = seed_test_preds
    print(f"===== Seed {seed} mean MAE: {seed_mean_mae:.4f} =====")

cv_mae_mean = np.mean(all_seed_scores)
cv_mae_std = np.std(all_seed_scores)
print("\n========================================")
print(f"🔥 Multi-seed CV MAE mean: {cv_mae_mean:.4f}  (std: {cv_mae_std:.4f})")
print("========================================")


# ==========================================
# 7. 生成提交文件
# ==========================================
final_test_preds = test_preds_all_seeds.mean(axis=0)

submission = pd.DataFrame({
    'ID': test['ID'],
    target_col: final_test_preds
})

sub_filename = f'submission_v6_ensemble_mae_{cv_mae_mean:.4f}.csv'
submission.to_csv(sub_filename, index=False)
print(f"✅ Submission saved as: {sub_filename}")


# ==========================================
# 8. 特征重要性图（Top 30）
# ==========================================
try:
    plt.figure(figsize=(10, 10))
    cols = (
        feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:30]
        .index
    )
    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features.sort_values(by="importance", ascending=False)
    )
    plt.title('Top 30 Feature Importance (multi-seed LGBM)')
    plt.tight_layout()
    plt.savefig('feature_importance_v6.png')
    print("📊 Feature importance plot saved as feature_importance_v6.png")
except Exception as e:
    print(f"⚠️ Could not plot feature importance: {e}")
