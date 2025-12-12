import pandas as pd

# 1. 锁定最强的两个文件
# LightGBM: V5.1 (LB 19.79 - 最强单模)
file_lgb = 'submission_v5_1_lgbm_mae_18.9639.csv'
# CatBoost: V7 (LB 19.88 - 最佳辅助)
file_cat = 'submission_v7_catboost_mae_19.0721.csv'

print(f"读取 LightGBM (V5.1): {file_lgb}")
df_lgb = pd.read_csv(file_lgb)

print(f"读取 CatBoost (V7): {file_cat}")
df_cat = pd.read_csv(file_cat)

target_col = 'fuel_consumption_sum'

# --- 策略 A: 40/60 (微调冠军策略) ---
# 既然 30/70 是 19.69，那我们试着往 LGB 这边挪一点点
# 理论依据：LGB 单模比 Cat 强，应该配得更高的权重
df_blend_4060 = df_lgb.copy()
df_blend_4060[target_col] = (df_lgb[target_col] * 0.40) + (df_cat[target_col] * 0.60)
file_4060 = 'submission_blend_40LGB_60Cat_V51.csv'
df_blend_4060.to_csv(file_4060, index=False)
print(f"✅ 生成: {file_4060} (40% V5.1 + 60% Cat)")

# --- 策略 B: 50/50 (强强联手) ---
# 两个都在 19.7-19.8 之间，平分秋色也许最好
df_blend_5050 = df_lgb.copy()
df_blend_5050[target_col] = (df_lgb[target_col] * 0.50) + (df_cat[target_col] * 0.50)
file_5050 = 'submission_blend_50LGB_50Cat_V5_1_7.csv'
df_blend_5050.to_csv(file_5050, index=False)
print(f"✅ 生成: {file_5050} (50% V5.1 + 50% Cat)")