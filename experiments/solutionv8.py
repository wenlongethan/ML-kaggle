import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


# ==========================================
# 1. 读取数据
# ==========================================
base_path = "/Users/liangwenlong/study/bme/3/ml_usage/project/truck-fuel-consumption-forecast"
train_path = os.path.join(base_path, "public_train.csv")
test_path = os.path.join(base_path, "public_test.csv")

print("🔹 Loading data ...")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target_col = "fuel_consumption_sum"


# ==========================================
# 2. Trip 级聚合特征 + 位置特征
# ==========================================
print("🔹 Building trip-level features ...")

trip_col = "Trip_ID_first"

# 合并，方便统一处理
train["is_train"] = 1
test["is_train"] = 0
df_all = pd.concat([train, test], axis=0, ignore_index=True)

# 按 trip + ID 排序，构造位置特征
if trip_col in df_all.columns and "ID" in df_all.columns:
    df_all = df_all.sort_values([trip_col, "ID"]).reset_index(drop=True)
    df_all["trip_len"] = df_all.groupby(trip_col)["ID"].transform("count")
    df_all["trip_pos"] = df_all.groupby(trip_col).cumcount()
    df_all["trip_pos_ratio"] = df_all["trip_pos"] / df_all["trip_len"].replace(0, 1)
else:
    df_all["trip_len"] = 1
    df_all["trip_pos"] = 0
    df_all["trip_pos_ratio"] = 0.0

# Trip 级聚合（不使用 label）
agg_dict = {
    "speed_mean": ["mean", "max", "min", "std"],
    "weight_1": ["mean"],
    "engine_percent_load_at_current_speed_mean": ["mean"],
}
agg_cols = [c for c in agg_dict.keys() if c in df_all.columns]
agg_dict = {k: v for k, v in agg_dict.items() if k in agg_cols}

trip_stats = df_all.groupby(trip_col).agg(agg_dict)
trip_stats.columns = [f"{c[0]}_{c[1]}_trip" for c in trip_stats.columns]
trip_stats = trip_stats.reset_index()

df_all = df_all.merge(trip_stats, on=trip_col, how="left")

# 物理特征
if {"weight_1", "speed_mean"}.issubset(df_all.columns):
    df_all["kinetic_energy"] = df_all["weight_1"] * (df_all["speed_mean"] ** 2)

# 风 + 船向（如果存在）
if "env_wind_kph" in df_all.columns:
    df_all["env_wind_kph"] = pd.to_numeric(df_all["env_wind_kph"], errors="coerce").fillna(0)
if "env_sailing_value" in df_all.columns:
    df_all["env_sailing_value"] = pd.to_numeric(df_all["env_sailing_value"], errors="coerce").fillna(0)
if {"env_wind_kph", "env_sailing_value"}.issubset(df_all.columns):
    df_all["wind_assist"] = df_all["env_wind_kph"] * df_all["env_sailing_value"]

# 拆回 train / test
train = df_all[df_all["is_train"] == 1].drop(columns=["is_train"]).reset_index(drop=True)
test = df_all[df_all["is_train"] == 0].drop(columns=["is_train"]).reset_index(drop=True)


# ==========================================
# 3. 真·OOF Target Encoding
# ==========================================
print("🔹 OOF target encoding ...")


def add_oof_target_encoding(train_df, test_df, cols, target, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for col in cols:
        if col not in train_df.columns:
            continue

        print(f"  -> OOF target encoding for {col}")
        oof = np.zeros(len(train_df))
        global_mean = train_df[target].mean()

        # OOF: 每个 fold 用当前 fold 的 train 部分计算均值，给 val 编码
        for tr_idx, val_idx in kf.split(train_df):
            tr = train_df.iloc[tr_idx]
            val = train_df.iloc[val_idx]
            mapping = tr.groupby(col)[target].mean()
            oof[val_idx] = val[col].map(mapping)

        # 填充缺失
        oof = np.where(np.isnan(oof), global_mean, oof)

        # test 用全训练集 mapping
        full_mapping = train_df.groupby(col)[target].mean()
        test_te = test_df[col].map(full_mapping).fillna(global_mean).values

        new_col = f"{col}_te"
        train_df[new_col] = oof
        test_df[new_col] = test_te

    return train_df, test_df


te_cols = ["driver_name_and_id", "vehicle_type", "route_id", "deviceuniquecode"]
train, test = add_oof_target_encoding(train, test, te_cols, target_col, n_splits=5, seed=42)

#（注意：原始类别列保留着，后面 features 里我们只用数值列，类别列当作给 NN 或后续实验用）


# ==========================================
# 4. LGBM + OOF 预测（为 stacking 做准备）
# ==========================================
drop_cols = ["ID", "Trip_ID_first", "Trip_ID_last", target_col]
features = [c for c in train.columns
            if c not in drop_cols and train[c].dtype != "object"]

print(f"🔹 Number of features: {len(features)}")

folds = 5
kf = KFold(n_splits=folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

print("🔹 Training LightGBM with trip features + OOF TE ...")

for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
    X_tr = train[features].iloc[tr_idx]
    y_tr = train[target_col].iloc[tr_idx]
    X_val = train[features].iloc[val_idx]
    y_val = train[target_col].iloc[val_idx]

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=40,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
    oof_preds[val_idx] = val_pred
    fold_mae = mean_absolute_error(y_val, val_pred)
    print(f"  Fold {fold + 1} MAE: {fold_mae:.4f}")

    test_preds += model.predict(test[features], num_iteration=model.best_iteration_) / folds

cv_mae = mean_absolute_error(train[target_col], oof_preds)
print(f"\n🔥 OOF CV MAE (LGBM + Trip + OOF TE): {cv_mae:.4f}")


# ==========================================
# 5. 保存 OOF & Submission（给 stacking / NN 用）
# ==========================================
oof_df = pd.DataFrame({
    "ID": train["ID"],
    "pred_lgb_trip": oof_preds,
    "y": train[target_col],
})
oof_df.to_csv("oof_lgb_trip_v8.csv", index=False)

sub_df = pd.DataFrame({
    "ID": test["ID"],
    target_col: test_preds,
})
sub_df.to_csv("submission_lgb_trip_v8.csv", index=False)

print("✅ Done. Saved oof_lgb_trip_v8.csv and submission_lgb_trip_v8.csv")
