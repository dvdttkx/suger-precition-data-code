import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# ============================== 全局工具函数区 ==============================
def get_stratified_folds(y, n_splits=5, seed=42):
    """按蛋白质含量分位数进行分层五折划分。
    若分位数分层失败（如重复值过多），则退回普通打乱五折。
    返回 [(train_idx, test_idx), ...] 列表，保证调参与最终评价使用同一套折。
    """
    try:
        y_bins = pd.qcut(y, q=n_splits, labels=False)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        X_dummy = np.arange(len(y))
        return list(skf.split(X_dummy, y_bins))
    except ValueError:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        X_dummy = np.arange(len(y))
        return list(kf.split(X_dummy))


def snv_transform(X):
    row_mean = np.mean(X, axis=1, keepdims=True)
    row_std = np.std(X, axis=1, ddof=1, keepdims=True)
    X_snv = (X - row_mean) / row_std
    return X_snv


def plsr_5fold(X, y, desc, n_comp=6, seed=42):
    rc2, rp2, rmsec, rmsep, rpd = [], [], [], [], []
    max_comp = min(X.shape[1], int(X.shape[0] * 0.8) - 1)
    use_comp = min(n_comp, max_comp)

    for tr_idx, te_idx in get_stratified_folds(y, seed=seed):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        pls = PLSRegression(n_components=use_comp)
        pls.fit(Xtr, ytr)
        y_cal = pls.predict(Xtr)
        y_pred = pls.predict(Xte)
        rc2.append(r2_score(ytr, y_cal))
        rp2.append(r2_score(yte, y_pred))
        rmsec.append(np.sqrt(mean_squared_error(ytr, y_cal)))
        rmsep.append(np.sqrt(mean_squared_error(yte, y_pred)))
        sd_te = np.std(yte, ddof=1)
        rpd.append(sd_te / rmsep[-1])

    res = {
        "方案": desc,
        "Rc2": np.mean(rc2), "Rp2": np.mean(rp2),
        "RMSEC": np.mean(rmsec), "RMSEP": np.mean(rmsep),
        "RPD": np.mean(rpd), "波段数": X.shape[1]
    }
    print(f"\n【{desc}】Rp2={res['Rp2']:.4f}, RPD={res['RPD']:.4f}")
    return res


def model_5fold(X, y, model, desc, seed=42):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    rc2, rp2, rmsec, rmsep, rpd = [], [], [], [], []

    for tr_idx, te_idx in get_stratified_folds(y, seed=seed):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        model.fit(Xtr, ytr)
        y_cal = model.predict(Xtr)
        y_pred = model.predict(Xte)
        rc2.append(r2_score(ytr, y_cal))
        rp2.append(r2_score(yte, y_pred))
        rmsec.append(np.sqrt(mean_squared_error(ytr, y_cal)))
        rmsep.append(np.sqrt(mean_squared_error(yte, y_pred)))
        sd_te = np.std(yte, ddof=1)
        rpd.append(sd_te / rmsep[-1])

    res = {
        "模型": desc,
        "Rc2": np.mean(rc2), "Rp2": np.mean(rp2),
        "RMSEC": np.mean(rmsec), "RMSEP": np.mean(rmsep),
        "RPD": np.mean(rpd)
    }
    print(f"\n【{desc}】Rp2={res['Rp2']:.4f}, RPD={res['RPD']:.4f}")
    return res


def vip_select(X, y, th=1.0, n_comp=6):
    pls = PLSRegression(n_components=n_comp).fit(X, y)
    W, T = pls.x_weights_, pls.x_scores_
    p = X.shape[1]
    vip = np.zeros(p)
    sum_t2 = np.sum(T**2, axis=0)
    for i in range(p):
        wi2 = W[i]**2
        vip[i] = np.sqrt(p * np.sum(sum_t2 * wi2) / np.sum(sum_t2))
    mask = vip > th
    return X[:, mask]


def cars_select(X, y, n_comp=6, iter_max=60, target_min=30, target_max=50):
    n_samp, n_var = X.shape
    mask = np.ones(n_var, bool)
    for _ in range(iter_max):
        now_X = X[:, mask]
        now_n = now_X.shape[1]
        if target_min <= now_n <= target_max or now_n <= target_min:
            break
        tr_num = int(n_samp * 0.8)
        tr_idx = np.random.choice(n_samp, tr_num, replace=False)
        pls = PLSRegression(n_components=n_comp).fit(now_X[tr_idx], y[tr_idx])
        w = np.abs(pls.x_weights_[:, 0])
        w /= w.sum()
        cut = int(now_n * 0.7)
        retain_idx = np.argsort(-w)[:cut]
        global_idx = np.where(mask)[0][retain_idx]
        mask[:] = False
        mask[global_idx] = True
    return X[:, mask]


def spa_select(X, y, max_num=80):
    from sklearn.linear_model import LinearRegression
    n_samp, n_var = X.shape
    sel = [np.random.randint(0, n_var)]
    for k in range(1, max_num):
        X_sel = X[:, sel]
        proj = X_sel @ np.linalg.inv(X_sel.T @ X_sel) @ X_sel.T
        x_proj = X - proj @ X
        norm = np.linalg.norm(x_proj, axis=0)
        cand = np.argmax(norm)
        if cand not in sel:
            sel.append(cand)
    best_r2 = -1
    best_sub = sel[:1]
    for num in range(1, len(sel) + 1):
        sub = sel[:num]
        lr = LinearRegression().fit(X[:, sub], y)
        r2 = r2_score(y, lr.predict(X[:, sub]))
        if r2 > best_r2:
            best_r2 = r2
            best_sub = sub
    return X[:, best_sub]


def mahalanobis_remove(X, y, threshold=3):
    X_mean = np.mean(X, axis=0)
    cov = np.cov(X.T)
    inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
    dists = []
    for row in X:
        diff = row - X_mean
        md = np.sqrt(diff @ inv_cov @ diff.T)
        dists.append(md)
    mask = np.array(dists) < threshold
    return X[mask], y[mask], mask.sum()


# ============================== 数据读取 ==============================
# 注：请根据实际文件名调整，这里使用“光谱原始数据集.xlsx”
file_path = r"D:\湿面筋特性预测\光谱原始数据集.xlsx"
df_raw = pd.read_excel(file_path)
y_all = df_raw["湿面筋特性平均"].values
wave_cols = [c for c in df_raw.columns if str(c).isdigit()]
X_raw_all = df_raw[wave_cols].values
print(f"原始数据：样本{X_raw_all.shape[0]}，波段{X_raw_all.shape[1]}")

# 统一的分层五折（调参与最终评价共用）
cv_folds = get_stratified_folds(y_all, n_splits=5, seed=42)


# ============================== 板块1：SNV+遍历SG窗口，PLSR分层五折 ==============================
print("\n============= 板块1：SNV预处理+SG窗口遍历 =============")
window_candidates = [9, 11, 13, 15, 17]
poly = 2
X_snv_base = snv_transform(X_raw_all)
res_win_list = []
for win in window_candidates:
    X_sg = savgol_filter(X_snv_base, window_length=win, polyorder=poly, deriv=0)
    res = plsr_5fold(X_sg, y_all, f"SNV+SG(win={win})")
    res_win_list.append(res)

df_win = pd.DataFrame(res_win_list).sort_values("RPD", ascending=False)
out_win = r"D:\湿面筋特性预测"
df_win.to_excel(out_win, index=False)
print("\n窗口排序结果：")
print(df_win)
best_win = int(df_win.iloc[0]["方案"].split("=")[-1].replace(")", ""))
print(f"板块1最优SG窗口：{best_win}")
X_snv_bestsg = savgol_filter(X_snv_base, window_length=best_win, polyorder=poly, deriv=0)


# ============================== 板块2：导数+三类特征筛选对比 ==============================
print("\n============= 板块2：导数+三类特征筛选对比 =============")
deriv_modes = [
    (0, "无导数"),
    (1, "一阶导数"),
    (2, "二阶导数")
]
filter_methods = [("VIP", vip_select), ("CARS", cars_select), ("SPA", spa_select)]
res_filter_list = []
for d_num, d_name in deriv_modes:
    X_der = savgol_filter(X_snv_bestsg, window_length=best_win, polyorder=poly, deriv=d_num)
    res_full = plsr_5fold(X_der, y_all, f"{d_name}_全波段")
    res_filter_list.append(res_full)
    for filt_name, filt_func in filter_methods:
        X_filt = filt_func(X_der, y_all)
        res_f = plsr_5fold(X_filt, y_all, f"{d_name}_{filt_name}")
        res_filter_list.append(res_f)

df_filt = pd.DataFrame(res_filter_list).sort_values("RPD", ascending=False)
out_filt = r"D:\湿面筋特性预测"
df_filt.to_excel(out_filt, index=False)
print("\n所有预处理+筛选组合排序：")
print(df_filt)

best_line = df_filt.iloc[0]
best_desc = best_line["方案"]
best_band_num = best_line["波段数"]
print(f"\n板块2最优组合：{best_desc}，保留波段{best_band_num}")

if "无导数" in best_desc:
    d_opt = 0
elif "一阶导数" in best_desc:
    d_opt = 1
else:
    d_opt = 2

X_opt_der = savgol_filter(X_snv_bestsg, window_length=best_win, polyorder=poly, deriv=d_opt)
if "VIP" in best_desc:
    X_opt_feat = vip_select(X_opt_der, y_all)
elif "CARS" in best_desc:
    X_opt_feat = cars_select(X_opt_der, y_all)
elif "SPA" in best_desc:
    X_opt_feat = spa_select(X_opt_der, y_all)
else:
    X_opt_feat = X_opt_der
print(f"最优特征集维度：{X_opt_feat.shape}")


# ============================== 板块3：多机器学习模型对比 ==============================
print("\n============= 板块3：多机器学习模型对比 =============")
model_pool = [
    (PLSRegression(n_components=6), "PLSR线性基准"),
    (RandomForestRegressor(random_state=42, n_estimators=100), "随机森林RF"),
    (XGBRegressor(random_state=42, n_estimators=100, objective="reg:squarederror"), "XGBoost"),
    (LGBMRegressor(random_state=42, n_estimators=100, verbose=-1), "LightGBM"),
    (CatBoostRegressor(random_state=42, verbose=False), "CatBoost")
]
res_model_list = []
X_mdf = pd.DataFrame(X_opt_feat)
for mod, m_name in model_pool:
    res_m = model_5fold(X_mdf, y_all, mod, m_name)
    res_model_list.append(res_m)

df_model = pd.DataFrame(res_model_list).sort_values("RPD", ascending=False)
out_model = r"D:\湿面筋特性预测"
df_model.to_excel(out_model, index=False)
print("\n模型排序：")
print(df_model)

candidate_model_info = [
    ("随机森林RF", RandomForestRegressor(random_state=42)),
    ("LightGBM", LGBMRegressor(random_state=42, verbose=-1)),
    ("CatBoost", CatBoostRegressor(random_state=42, verbose=False))
]


# ============================== 板块4：候选模型全部网格超参数调优（统一分层五折） ==============================
print("\n============= 板块4：RF / LightGBM / CatBoost 分别网格调参 =============")
X_grid = pd.DataFrame(X_opt_feat)
tune_result_list = []

param_grid_dict = {
    "随机森林RF": {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "min_samples_split": [2, 3]
    },
    "LightGBM": {
        "n_estimators": [50, 100],
        "max_depth": [2, 3],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [4, 8],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [1, 5]
    },
    "CatBoost": {
        "iterations": [50, 100],
        "depth": [2, 3],
        "learning_rate": [0.05, 0.1]
    }
}

trained_models = dict()

for name, base_estimator in candidate_model_info:
    print(f"\n-------- 正在调优 {name} --------")
    grid_p = param_grid_dict[name]
    # 关键修改：GridSearchCV 使用与 model_5fold 完全相同的分层五折
    gs = GridSearchCV(base_estimator, grid_p, cv=cv_folds, scoring="r2", n_jobs=1)
    gs.fit(X_grid, y_all)
    best_para = gs.best_params_
    print(f"{name} 最优超参数: {best_para}")
    print(f"{name} 网格交叉最优R2: {gs.best_score_:.4f}")

    if name == "随机森林RF":
        opt_mod = RandomForestRegressor(random_state=42, **best_para)
    elif name == "LightGBM":
        opt_mod = LGBMRegressor(random_state=42, verbose=-1, **best_para)
    elif name == "CatBoost":
        opt_mod = CatBoostRegressor(random_state=42, verbose=False, **best_para)
    else:
        opt_mod = base_estimator

    trained_models[name] = {"model": opt_mod, "params": best_para}
    res_tune = model_5fold(X_grid, y_all, opt_mod, f"调优后_{name}")
    tune_result_list.append(res_tune)

df_tune_all = pd.DataFrame(tune_result_list)
out_tune = r"D:\湿面筋特性预测"
df_tune_all.to_excel(out_tune, index=False)
print("\n三模型调参后指标汇总：")
print(df_tune_all)


# ============================== 板块5：每个模型分别对比【全样本】vs【马氏距离剔除异常样本】 ==============================
print("\n============= 板块5：全样本 VS 剔除异常样本对比（RF / LightGBM / CatBoost） =============")
X_clean, y_clean, retain_num = mahalanobis_remove(X_opt_feat, y_all, threshold=3)
if retain_num < 2:
    print("剔除后样本不足，使用全样本替代")
    X_clean, y_clean = X_opt_feat, y_all
    retain_num = len(y_all)
X_clean_df = pd.DataFrame(X_clean)

abnormal_compare_list = []
for name, info in trained_models.items():
    mod = info["model"]
    res_full = model_5fold(X_grid, y_all, mod, f"{name}_全样本")
    res_out = model_5fold(X_clean_df, y_clean, mod, f"{name}_剔除异常")
    abnormal_compare_list.append(res_full)
    abnormal_compare_list.append(res_out)

res_abnormal = pd.DataFrame(abnormal_compare_list)
out_ab = r"D:\湿面筋特性预测"
res_abnormal.to_excel(out_ab, index=False)
print(f"\n原始样本:{len(y_all)}，剔除异常后保留样本:{retain_num}")
print("\n异常剔除前后全部对比表：")
print(res_abnormal)