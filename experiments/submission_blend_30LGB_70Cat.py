import pandas as pd

# 1. 再次确认文件名 (请确保这两个文件都在文件夹里)
lgb_file = 'submission_v5_1_lgbm_mae_18.9639.csv' # 或者用 v6_robust (19.15那个)
cat_file = 'submission_v7_catboost_mae_19.0721.csv'

print(f"读取 LightGBM: {lgb_file}")
df_lgb = pd.read_csv(lgb_file)

print(f"读取 CatBoost: {cat_file}")
df_cat = pd.read_csv(cat_file)

# 2. 生成加权融合 (Weighted Blend)
target_col = 'fuel_consumption_sum'

# 方案 A: 偏向 CatBoost (推荐)
# 30% LGBM + 70% CatBoost
df_blend_7030 = df_lgb.copy()
df_blend_7030[target_col] = (df_lgb[target_col] * 0.3) + (df_cat[target_col] * 0.7)
file_7030 = 'submission_blend_30LGB_70Cat.csv'
df_blend_7030.to_csv(file_7030, index=False)
print(f"✅ 生成文件: {file_7030} (策略: 30% LGBM + 70% Cat)")

# 方案 B: 激进版 (CatBoost 主导)
# 15% LGBM + 85% CatBoost
df_blend_8515 = df_lgb.copy()
df_blend_8515[target_col] = (df_lgb[target_col] * 0.15) + (df_cat[target_col] * 0.85)
file_8515 = 'submission_blend_15LGB_85Cat.csv'
df_blend_8515.to_csv(file_8515, index=False)
print(f"✅ 生成文件: {file_8515} (策略: 15% LGBM + 85% Cat)")