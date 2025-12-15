import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 读入数据
# ==========================================
base_path = "/Users/liangwenlong/study/bme/3/ml_usage/project/truck-fuel-consumption-forecast"
train_path = os.path.join(base_path, "public_train.csv")
test_path  = os.path.join(base_path, "public_test.csv")

print("正在读取数据...")
train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

target_col = "fuel_consumption_sum"

# 为了后面生成 submission，用一个单独变量保存 ID
test_ids = test["ID"].copy()

# ==========================================
# 2. 基础特征工程：物理 & 环境交互特征
# ==========================================
print("正在构建基础物理特征...")

for df in [train, test]:
    # 动能与动量
    df["kinetic_energy"] = df["weight_1"] * (df["speed_mean"] ** 2)
    df["momentum"] = df["weight_1"] * df["speed_mean"]

    # 功率需求
    df["power_demand"] = df["engine_percent_load_at_current_speed_mean"] * df["engine_speed_mean"]

    # 环境风 vs 航向
    if "env_wind_kph" in df.columns and "env_sailing_value" in df.columns:
        df["env_wind_kph"] = pd.to_numeric(df["env_wind_kph"], errors="coerce")
        df["env_sailing_value"] = pd.to_numeric(df["env_sailing_value"], errors="coerce")
        df["wind_assist"] = df["env_wind_kph"].fillna(0) * df["env_sailing_value"].fillna(0)

# ==========================================
# 3. OOF Target Encoding（防止目标泄露）
# ==========================================
def target_encode_oof(train_df, test_df, col, target, n_splits=5, seed=42):
    """
    对某一列做 Out-of-Fold target encoding：
    - 每一折的编码只用「其他折」的标签统计
    - test 用全体 train 统计
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    te_train = pd.Series(index=train_df.index, dtype=float)

    for tr_idx, va_idx in kf.split(train_df):
        tr_fold = train_df.iloc[tr_idx]
        mapping = tr_fold.groupby(col)[target].mean()
        te_train.iloc[va_idx] = train_df[col].iloc[va_idx].map(mapping)

    # test：用全体 train 的统计
    global_mapping = train_df.groupby(col)[target].mean()
    te_test = test_df[col].map(global_mapping)

    global_mean = train_df[target].mean()
    te_train = te_train.fillna(global_mean)
    te_test = te_test.fillna(global_mean)

    return te_train, te_test


print("正在进行 Target Encoding (OOF)...")

# 根据你之前的选择，这些是高价值类别列
te_cols = ["driver_name_and_id",
           "vehicle_type",
           "route_id",
           "vehicle_motortype",
           "deviceuniquecode"]

existing_te_cols = [c for c in te_cols if c in train.columns]
print(f"将进行 OOF Target Encoding 的列: {existing_te_cols}")

for col in existing_te_cols:
    train[f"{col}_te"], test[f"{col}_te"] = target_encode_oof(
        train, test, col, target_col, n_splits=5, seed=42
    )

# ==========================================
# 4. 特征筛选 & 类别列指定
#    - 原始高基数 ID / code / name 列全部丢弃，只保留 *_te
# ==========================================
drop_cols = [
    "ID", "Trip_ID_first", "Trip_ID_last", target_col,
    "driver_name_and_id", "vehicle_type", "route_id",
    "vehicle_motortype", "deviceuniquecode"
]

features = [c for c in train.columns if c not in drop_cols]

# 指定类别列：只对真正的字符串枚举类型生效
cat_cols = []
for col in features:
    if train[col].dtype == "object":
        train[col] = train[col].astype("category")
        test[col] = test[col].astype("category")
        cat_cols.append(col)

print(f"最终特征数: {len(features)}")
print(f"类别特征: {cat_cols}")

# ==========================================
# 5. LightGBM 5-fold CV，目标 = MAE
# ==========================================
folds = 5
kf = KFold(n_splits=folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_maes = []
feature_importance_df = pd.DataFrame()

params = {
    "objective": "mae",     # 直接优化 MAE
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
    "n_jobs": -1,
    "seed": 42
}

print("开始训练 LightGBM (V5.1 - MAE, OOF-TE)...")

for fold, (tr_idx, va_idx) in enumerate(kf.split(train)):
    print(f"\n========== Fold {fold+1}/{folds} ==========")
    X_tr, y_tr = train[features].iloc[tr_idx], train[target_col].iloc[tr_idx]
    X_va, y_va = train[features].iloc[va_idx], train[target_col].iloc[va_idx]

    dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, y_va, categorical_feature=cat_cols, free_raw_data=False)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(200),
            lgb.log_evaluation(500)
        ]
    )

    # 使用 best_iteration 做预测，和早停保持一致
    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[va_idx] = val_pred
    test_preds += model.predict(
        test[features],
        num_iteration=model.best_iteration
    ) / folds

    mae = mean_absolute_error(y_va, val_pred)
    fold_maes.append(mae)
    print(f"Fold {fold+1} MAE: {mae:.4f}")

    # 记录特征重要性（gain 更有意义）
    fold_importance = pd.DataFrame({
        "feature": features,
        "importance_gain": model.feature_importance(importance_type="gain"),
        "importance_split": model.feature_importance(importance_type="split"),
        "fold": fold + 1
    })
    feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)

# ==========================================
# 6. 结果 & 提交文件
# ==========================================
mean_mae = float(np.mean(fold_maes))
print("\n========================================")
print(f"🔥 V5.1 (MAE + OOF-TE) CV Mean MAE: {mean_mae:.4f}")
print("========================================")

submission = pd.DataFrame({
    "ID": test_ids,
    target_col: test_preds
})

sub_filename = f"submission_v5_1_lgbm_mae_{mean_mae:.4f}.csv"
submission.to_csv(sub_filename, index=False)
print(f"提交文件已生成: {sub_filename}")

# ==========================================
# 7. 绘制 Top-20 特征重要性
# ==========================================
plt.figure(figsize=(10, 8))
cols = (feature_importance_df[["feature", "importance_gain"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance_gain", ascending=False)[:20].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

sns.barplot(
    x="importance_gain",
    y="feature",
    data=best_features.sort_values(by="importance_gain", ascending=False)
)
plt.title("Top 20 Features (V5.1 - MAE + OOF-TE)")
plt.tight_layout()
plt.savefig("feature_importance_v5_1.png")
print("特征重要性图已保存为 feature_importance_v5_1.png")
