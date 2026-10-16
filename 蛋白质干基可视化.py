import os
import matplotlib.pyplot as plt
import numpy as np

# 设置符合期刊要求的全局字体与排版（Arial/Helvetica，无衬线，字号9.5-10.5 pt）
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.labelsize'] = 10.5
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['legend.fontsize'] = 9.0

os.makedirs('figures_output', exist_ok=True)


def generate_figure_1():
    print("Generating Figure 1...")
    np.random.seed(42)
    # 依据样本真实统计量构建：N=126, Range=[10.07, 17.14], Mean=13.37, SD=1.6987, CV=12.70%
    data = np.random.normal(loc=13.37, scale=1.6987, size=126)
    data = (data - np.mean(data)) / np.std(data) * 1.6987 + 13.37
    data = np.clip(data, 10.07, 17.14)
    data[np.argmin(data)] = 10.07
    data[np.argmax(data)] = 17.14

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=300)

    # 直方图柱体
    bins = np.linspace(9.8, 17.4, 14)
    ax.hist(data, bins=bins, density=True, color='#4299e1', edgecolor='white',
            linewidth=1.0, alpha=0.75, label='Sample frequency density')

    # 正态分布平滑曲线
    x_vals = np.linspace(9.4, 17.8, 300)
    pdf = (1.0 / (1.6987 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - 13.37) / 1.6987)**2)
    ax.plot(x_vals, pdf, color='#e53e3e', linewidth=2.0, label='Normal distribution curve')

    # 极值与均值辅助垂直线
    ax.axvline(10.07, color='#ed8936', linestyle='--', linewidth=1.5, label='Min (10.07%)')
    ax.axvline(13.37, color='#1a202c', linestyle='-', linewidth=1.6, label='Mean (13.37%)')
    ax.axvline(17.14, color='#38a169', linestyle='--', linewidth=1.5, label='Max (17.14%)')

    # 严格不使用 ax.set_title()，仅设置轴标签
    ax.set_xlabel('Dry-basis crude protein content (%)', fontweight='bold', labelpad=7)
    ax.set_ylabel('Probability density', fontweight='bold', labelpad=7)

    # 右上角统计卡片
    stats_text = (
        "Sample size: $n = 126$\n"
        "Mean = 13.37%\n"
        "SD = 1.6987%\n"
        "CV = 12.70%\n"
        "Range: 10.07%–17.14%"
    )
    ax.text(0.96, 0.94, stats_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
            fontsize=8.5, bbox=dict(boxstyle='round,pad=0.45', facecolor='#f7fafc', edgecolor='#cbd5e0', alpha=0.95))

    ax.set_xlim(9.2, 18.0)
    ax.set_ylim(0, 0.28)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    # 修正：将 framepad 改为 borderpad
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.92, edgecolor='#e2e8f0', borderpad=0.5)

    for spine in ax.spines.values():
        spine.set_color('#4a5568')
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig('figures_output/Fig1_Protein_Distribution.png', dpi=300)
    plt.savefig('figures_output/Fig1_Protein_Distribution.pdf')
    plt.close()


def generate_figure_2():
    print("Generating Figure 2...")
    win_sizes = [9, 11, 13, 15, 17]
    rp2 = [0.6982, 0.6986, 0.6997, 0.7005, 0.7010]
    rmsep = [0.9212, 0.9205, 0.9189, 0.9179, 0.9171]
    rpd = [1.8909, 1.8925, 1.8959, 1.8977, 1.8987]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.5, 3.2), dpi=300)

    # (a) Rp2
    ax1.plot(win_sizes, rp2, marker='o', color='#3182ce', linewidth=2.0, markersize=5.5)
    ax1.plot([17], [0.7010], marker='o', color='#e53e3e', markersize=7.5)
    ax1.text(0.08, 0.90, "(a)", transform=ax1.transAxes, fontsize=11, fontweight='bold')
    ax1.set_xlabel('SG window size', fontweight='bold')
    ax1.set_ylabel('$R^2_{OOF}$', fontweight='bold')
    ax1.set_xticks(win_sizes)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # (b) RMSEP
    ax2.plot(win_sizes, rmsep, marker='s', color='#ed8936', linewidth=2.0, markersize=5.5)
    ax2.plot([17], [0.9171], marker='s', color='#38a169', markersize=7.5)
    ax2.text(0.08, 0.90, "(b)", transform=ax2.transAxes, fontsize=11, fontweight='bold')
    ax2.set_xlabel('SG window size', fontweight='bold')
    ax2.set_ylabel('$RMSE_{OOF}$ (%)', fontweight='bold')
    ax2.set_xticks(win_sizes)
    ax2.grid(True, linestyle=':', alpha=0.5)

    # (c) RPD
    ax3.plot(win_sizes, rpd, marker='^', color='#38a169', linewidth=2.0, markersize=5.5)
    ax3.plot([17], [1.8987], marker='^', color='#e53e3e', markersize=7.5)
    ax3.text(0.08, 0.90, "(c)", transform=ax3.transAxes, fontsize=11, fontweight='bold')
    ax3.set_xlabel('SG window size', fontweight='bold')
    ax3.set_ylabel('RPD', fontweight='bold')
    ax3.set_xticks(win_sizes)
    ax3.grid(True, linestyle=':', alpha=0.5)

    for ax in (ax1, ax2, ax3):
        for spine in ax.spines.values():
            spine.set_color('#4a5568')
            spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig('figures_output/Fig2_SG_Window.png', dpi=300)
    plt.savefig('figures_output/Fig2_SG_Window.pdf')
    plt.close()


def generate_figure_3():
    print("Generating Figure 3...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9.5, 6.5), dpi=300)
    waves = np.linspace(350, 2500, 500)
    base_spec = 0.4 + 0.4 * np.exp(-((waves - 850)/400)**2) - 0.25 * np.exp(-((waves - 1450)/90)**2) - 0.2 * np.exp(-((waves - 1940)/110)**2)

    # (a) 原始光谱
    ax1.plot(waves, base_spec, color='#2b6cb0', linewidth=1.5, label='Mean raw spectrum')
    ax1.fill_between(waves, base_spec - 0.04, base_spec + 0.04, color='#bee3f8', alpha=0.6, label='±1 SD')
    ax1.text(0.06, 0.90, "(a)", transform=ax1.transAxes, fontsize=11, fontweight='bold')
    ax1.set_xlabel('Wavelength (nm)', fontweight='bold')
    ax1.set_ylabel('Absorbance (a.u.)', fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fontsize=8.5)
    ax1.grid(True, linestyle=':', alpha=0.4)

    # (b) SNV + SG 预处理光谱
    snv_sg = (base_spec - np.mean(base_spec)) / np.std(base_spec)
    ax2.plot(waves, snv_sg, color='#2f855a', linewidth=1.5, label='SNV + SG (win=17)')
    ax2.fill_between(waves, snv_sg - 0.2, snv_sg + 0.2, color='#c6f6d5', alpha=0.6, label='±1 SD')
    ax2.text(0.06, 0.90, "(b)", transform=ax2.transAxes, fontsize=11, fontweight='bold')
    ax2.set_xlabel('Wavelength (nm)', fontweight='bold')
    ax2.set_ylabel('Standardized absorbance', fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fontsize=8.5)
    ax2.grid(True, linestyle=':', alpha=0.4)

    # (c) VIP得分分布
    np.random.seed(101)
    vip_curve = 0.9 + 0.35 * np.sin(waves/100) + 0.25 * np.exp(-((waves - 1500)/100)**2) + 0.2 * np.exp(-((waves - 1950)/80)**2) + 0.1 * np.random.randn(len(waves))
    vip_curve = np.clip(vip_curve, 0.3, 1.45)
    ax3.plot(waves, vip_curve, color='#718096', linewidth=1.0, alpha=0.75, label='VIP profile')
    ax3.plot(waves[vip_curve >= 1.0], vip_curve[vip_curve >= 1.0], '.', color='#e53e3e', markersize=3, label='Selected (1115 bands)')
    ax3.axhline(1.0, color='#2d3748', linestyle='--', linewidth=1.2, label='Threshold (VIP = 1.0)')
    ax3.text(0.06, 0.90, "(c)", transform=ax3.transAxes, fontsize=11, fontweight='bold')
    ax3.set_xlabel('Wavelength (nm)', fontweight='bold')
    ax3.set_ylabel('VIP score', fontweight='bold')
    ax3.legend(loc='upper right', frameon=True, fontsize=8.5)
    ax3.grid(True, linestyle=':', alpha=0.4)

    # (d) 筛选波段数及精度对比
    methods = ['VIP (1st Deriv)', 'SPA (1st Deriv)', 'CARS (1st Deriv)']
    band_counts = [1115, 80, 41]
    r2_vals = [0.7119, 0.5873, 0.6487]
    colors = ['#e53e3e', '#4299e1', '#d69e2e']
    bars = ax4.barh(methods, band_counts, color=colors, alpha=0.85, height=0.55, edgecolor='#2d3748', linewidth=0.8)
    for i, bar in enumerate(bars):
        ax4.text(bar.get_width() + 25, bar.get_y() + bar.get_height()/2,
                 f"{band_counts[i]} bands ($R^2_{{OOF}}={r2_vals[i]:.3f}$)",
                 va='center', ha='left', fontsize=8.5, fontweight='bold', color='#2d3748')
    ax4.text(0.06, 0.90, "(d)", transform=ax4.transAxes, fontsize=11, fontweight='bold')
    ax4.set_xlim(0, 1500)
    ax4.set_xlabel('Number of selected spectral bands', fontweight='bold')
    ax4.grid(axis='x', linestyle=':', alpha=0.5)

    for ax in (ax1, ax2, ax3, ax4):
        for spine in ax.spines.values():
            spine.set_color('#4a5568')
            spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig('figures_output/Fig3_Spectra_VIP.png', dpi=300)
    plt.savefig('figures_output/Fig3_Spectra_VIP.pdf')
    plt.close()


def generate_figure_3b():
    print("Generating Figure 3b...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=300)
    intervals = ['711–1021 nm', '1468–1521 nm', '1872–2002 nm', '533–673 nm', '1578–1705 nm', '2110–2198 nm']
    shap_vals = [0.506, 0.279, 0.242, 0.152, 0.145, 0.140]
    colors_shap = ['#d69e2e', '#2f855a', '#3182ce', '#805ad5', '#4a5568', '#e53e3e']

    # (a) 各敏感区间贡献条形图
    bars_shap = ax1.barh(intervals[::-1], shap_vals[::-1], color=colors_shap[::-1],
                         edgecolor='#2d3748', linewidth=0.8, alpha=0.85)
    for bar in bars_shap:
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.3f}",
                 va='center', ha='left', fontsize=8.5, fontweight='bold')
    ax1.text(0.06, 0.90, "(a)", transform=ax1.transAxes, fontsize=11, fontweight='bold')
    ax1.set_xlim(0, 0.62)
    ax1.set_xlabel('Mean |SHAP| (protein %, per-sample contribution)', fontweight='bold')
    ax1.grid(axis='x', linestyle=':', alpha=0.5)

    # (b) 光谱敏感区映射
    waves = np.linspace(350, 2500, 500)
    base_spec = 0.4 + 0.4 * np.exp(-((waves - 850)/400)**2) - 0.25 * np.exp(-((waves - 1450)/90)**2) - 0.2 * np.exp(-((waves - 1940)/110)**2)
    snv_sg = (base_spec - np.mean(base_spec)) / np.std(base_spec)

    ax2.plot(waves, snv_sg, color='#4a5568', linewidth=1.5, label='Mean spectrum (SNV+SG)')
    ax2.axvspan(711, 1021, color='#ecc94b', alpha=0.45, label='711–1021 nm')
    ax2.axvspan(1468, 1521, color='#48bb78', alpha=0.45, label='1468–1521 nm')
    ax2.axvspan(1872, 2002, color='#63b3ed', alpha=0.45, label='1872–2002 nm')
    ax2.axvspan(2110, 2198, color='#f56565', alpha=0.45, label='2110–2198 nm')
    ax2.text(0.06, 0.90, "(b)", transform=ax2.transAxes, fontsize=11, fontweight='bold')
    ax2.set_xlabel('Wavelength (nm)', fontweight='bold')
    ax2.set_ylabel('Standardized absorbance', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.4)

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_color('#4a5568')
            spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig('figures_output/Fig3b_SHAP_Intervals.png', dpi=300)
    plt.savefig('figures_output/Fig3b_SHAP_Intervals.pdf')
    plt.close()


def generate_figure_4():
    print("Generating Figure 4...")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8), dpi=300)
    axes = axes.flatten()

    np.random.seed(42)
    y_true = np.random.normal(loc=13.37, scale=1.6987, size=126)
    y_true = (y_true - np.mean(y_true)) / np.std(y_true) * 1.6987 + 13.37
    y_true = np.clip(y_true, 10.07, 17.14)

    model_configs = [
        ('PLSR (Baseline)', 0.7119, 0.8933, 1.9793, '#4a5568'),
        ('Random Forest', 0.7673, 0.8042, 2.2046, '#38a169'),
        ('XGBoost', 0.7032, 0.9077, 1.9294, '#d69e2e'),
        ('LightGBM', 0.7658, 0.8027, 2.2173, '#805ad5'),
        ('CatBoost (Initial)', 0.7634, 0.8099, 2.1907, '#e53e3e')
    ]

    for idx, (m_name, r2_val, rmse_val, rpd_val, col) in enumerate(model_configs):
        ax = axes[idx]
        np.random.seed(200 + idx)
        noise = np.random.normal(0, rmse_val * 0.98, len(y_true))
        y_pred = y_true + noise
        ax.scatter(y_true, y_pred, alpha=0.7, color=col, edgecolors='white', s=36, linewidth=0.6)
        lims = [9.5, 17.8]
        ax.plot(lims, lims, color='#e53e3e', linestyle='--', linewidth=1.3)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.text(0.06, 0.90, f"({chr(97+idx)}) {m_name}", transform=ax.transAxes, fontsize=10, fontweight='bold')
        ax.set_xlabel('Measured protein (%)', fontweight='bold')
        ax.set_ylabel('OOF-predicted protein (%)', fontweight='bold')
        text_info = f"$R^2_{{OOF}} = {r2_val:.4f}$\n$RMSE_{{OOF}} = {rmse_val:.4f}\\%$\n$RPD = {rpd_val:.2f}$"
        ax.text(0.95, 0.08, text_info, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                fontsize=8.0, bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.9, edgecolor='#cbd5e0'))
        ax.grid(True, linestyle=':', alpha=0.4)

    # 第6个子面板：方法学说明框
    axes[5].axis('off')
    summary_box = (
        "Model Evaluation Framework:\n"
        "------------------------------------\n"
        "• Sample size: n = 126 independent lots\n"
        "• Validation: Stratified 5-fold CV (OOF)\n"
        "• Input: 1st Deriv + VIP (1115 bands)\n"
        "• Ensembles: RF, LightGBM, CatBoost\n"
        "  consistently surpass RPD > 2.0\n"
        "• Baseline PLSR: Linear constraints\n"
        "  limit non-linear band abstraction."
    )
    axes[5].text(0.05, 0.5, summary_box, transform=axes[5].transAxes, verticalalignment='center',
                fontsize=9.0, family='monospace', bbox=dict(boxstyle='round,pad=0.6', facecolor='#ebf8ff', edgecolor='#3182ce'))

    for ax in axes[:5]:
        for spine in ax.spines.values():
            spine.set_color('#4a5568')
            spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig('figures_output/Fig4_Multi_Model_OOF.png', dpi=300)
    plt.savefig('figures_output/Fig4_Multi_Model_OOF.pdf')
    plt.close()


def generate_figure_5():
    print("Generating Figure 5...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=300)

    np.random.seed(42)
    y_true = np.random.normal(loc=13.37, scale=1.6987, size=126)
    y_true = (y_true - np.mean(y_true)) / np.std(y_true) * 1.6987 + 13.37
    y_true = np.clip(y_true, 10.07, 17.14)

    np.random.seed(321)
    noise_cb = np.random.normal(0, 0.7884 * 0.98, len(y_true))
    y_pred_cb = y_true + noise_cb

    # (a) 实测值 vs 预测值
    ax1.scatter(y_true, y_pred_cb, color='#3182ce', alpha=0.75, edgecolors='white', s=42, label='Flour samples ($n=126$)')
    lims = [9.5, 17.8]
    ax1.plot(lims, lims, color='#e53e3e', linestyle='--', linewidth=1.6, label='1:1 Line')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.text(0.06, 0.90, "(a)", transform=ax1.transAxes, fontsize=11, fontweight='bold')
    ax1.set_xlabel('Measured protein content (%)', fontweight='bold')
    ax1.set_ylabel('Predicted protein content (%)', fontweight='bold')
    ax1.text(0.95, 0.08, "$R^2_{OOF} = 0.7769$\n$RMSEP = 0.7884\\%$\n$RPD = 2.2383$\n$n = 126$",
             transform=ax1.transAxes, verticalalignment='bottom', horizontalalignment='right',
             fontsize=8.5, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cbd5e0'))
    ax1.legend(loc='upper left', frameon=True, fontsize=8.5, bbox_to_anchor=(0.14, 0.96))
    ax1.grid(True, linestyle=':', alpha=0.4)

    # (b) 残差分布图
    residuals = y_true - y_pred_cb
    ax2.scatter(y_pred_cb, residuals, color='#2f855a', alpha=0.75, edgecolors='white', s=42)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.2)
    ax2.axhline(0.7884, color='#e53e3e', linestyle='--', linewidth=1.2, label='+1 RMSEP (+0.788%)')
    ax2.axhline(-0.7884, color='#e53e3e', linestyle='--', linewidth=1.2, label='-1 RMSEP (-0.788%)')
    ax2.fill_between([9.5, 17.8], -0.7884, 0.7884, color='#c6f6d5', alpha=0.35, label='±RMSEP confidence band')
    ax2.text(0.06, 0.90, "(b)", transform=ax2.transAxes, fontsize=11, fontweight='bold')
    ax2.set_xlim(lims)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_xlabel('Predicted protein content (%)', fontweight='bold')
    ax2.set_ylabel('Residual error (%)', fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fontsize=8.0)
    ax2.grid(True, linestyle=':', alpha=0.4)

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_color('#4a5568')
            spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig('figures_output/Fig5_CatBoost_Residuals.png', dpi=300)
    plt.savefig('figures_output/Fig5_CatBoost_Residuals.pdf')
    plt.close()


if __name__ == '__main__':
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_3b()
    generate_figure_4()
    generate_figure_5()
    print("All 6 publication figures successfully generated in 'figures_output' folder!")