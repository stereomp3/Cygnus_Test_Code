"""
使用 marker 紀錄位置，用 marker 比較
預設實驗，敲某個 channel，測試
看看一開始是否會有 LSL 阻塞問題
8.298-1.962 = 6.336
17:53:36.209-17:53:29.823497 = 36.209-29.823 = 6.386
"""
# compare_with_marker_post2s.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from scipy.signal import correlate


# ---- helper functions ----
def read_cygnus_csv(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        lines = f.read().splitlines()
    header_idx = next(i for i, l in enumerate(lines) if l.strip().startswith("Timestamp"))
    meta_lines = lines[:header_idx]
    data_lines = lines[header_idx:]
    df = pd.read_csv(StringIO("\n".join(data_lines)))
    df.columns = df.columns.str.strip()
    rec_line = next((l for l in meta_lines if 'Record datetime' in l), None)
    if rec_line is None:
        raise ValueError("找不到 'Record datetime' 行於 cygnus 檔案")
    rec_str = rec_line.split(':', 1)[1].strip()
    rec_dt = pd.to_datetime(rec_str)
    return df, rec_dt


def read_realtime_csv(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        lines = f.read().splitlines()
    rec_line = lines[0]
    if 'Record datetime' not in rec_line:
        raise ValueError("realtime 檔案第一行找不到 'Record datetime'")
    rec_str = rec_line.split(':', 1)[1].strip()
    rec_dt = pd.to_datetime(rec_str)
    data_lines = lines[1:]
    df = pd.read_csv(StringIO("\n".join(data_lines)))
    df.columns = df.columns.str.strip()
    return df, rec_dt


def find_lag(y_c, y_r):
    corr = correlate(y_c - np.mean(y_c), y_r - np.mean(y_r), mode="full")
    lags = np.arange(-len(y_c) + 1, len(y_c))
    lag = lags[np.argmax(corr)]
    return lag


def overlay_plot(ch, y_c, y_r, t, align=False, n_samples=None, is_plot=False, ax=None):
    if align:
        lag = find_lag(y_c, y_r)
        if lag > 0:
            y_r = np.concatenate([np.full(lag, np.nan), y_r[:-lag]])
        elif lag < 0:
            y_r = np.concatenate([y_r[-lag:], np.full(-lag, np.nan)])
        print(f"{ch} 對齊 lag = {lag} samples (~{lag / 500:.3f} 秒)。real-time shift.")

    if n_samples:
        y_c = y_c[:n_samples]
        y_r = y_r[:n_samples]
        t = t[:n_samples]

    diff = y_c - y_r
    if is_plot and ax is not None:
        ax.plot(t, y_c, label=f'{ch} cy')
        ax.plot(t, y_r, label=f'{ch} rt', alpha=0.8)
        ax.plot(t, diff, label='diff', linestyle='--')
        ax.set_title(f'{ch}')
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='x-small')
    return diff


# ---- main ----
cygnus_path = "data/250914175336.csv"
realtime_path = "data/eeg_record.csv"
marker_path = "data/markers.csv"

df_cy, dt_cy = read_cygnus_csv(cygnus_path)
df_rt, dt_rt = read_realtime_csv(realtime_path)
df_marker = pd.read_csv(marker_path)

# 清理欄位
df_cy = df_cy.loc[:, ~df_cy.columns.str.contains('^Unnamed')]
df_rt = df_rt.loc[:, ~df_rt.columns.str.contains('^Unnamed')]

df_cy['Timestamp'] = pd.to_numeric(df_cy['Timestamp'], errors='coerce')
df_rt['Timestamp'] = pd.to_numeric(df_rt['Timestamp'], errors='coerce')
df_cy = df_cy.dropna(subset=['Timestamp']).reset_index(drop=True)
df_rt = df_rt.dropna(subset=['Timestamp']).reset_index(drop=True)

df_cy['abs_time'] = dt_cy + pd.to_timedelta(df_cy['Timestamp'], unit='s')
df_rt['abs_time'] = dt_rt + pd.to_timedelta(df_rt['Timestamp'], unit='s')

# 找出共同 channels
exclude = {'Timestamp', 'Serial Number', 'Event Id', 'Event Date', 'Event Duration',
           'Software Marker', 'Software Marker Name', 'abs_time'}
channels = [c for c in df_cy.columns if c in df_rt.columns and c not in exclude]
print("偵測到要比對的 channels:", channels)

# ---- 針對每個 marker 抽取往後 2 秒 ----
segment_results = {}

for _, row in df_marker.iterrows():
    marker_time = float(row["Timestamp"])
    marker_label = row["Marker label"]

    print(f"\n=== Marker {marker_label} ===")

    # Realtime：用 marker.csv 的 timestamp
    rt_mask = (df_rt["Timestamp"] >= marker_time) & (df_rt["Timestamp"] <= marker_time + 2)
    seg_rt = df_rt.loc[rt_mask].reset_index(drop=True)

    # Cygnus：用 Software Marker
    cy_mask_marker = df_cy["Software Marker"] == marker_label
    if not cy_mask_marker.any():
        print(f"⚠️ Cygnus 找不到 Software Marker {marker_label}，跳過。")
        continue

    marker_time_cy = df_cy.loc[cy_mask_marker, "Timestamp"].iloc[0]
    cy_mask = (df_cy["Timestamp"] >= marker_time_cy) & (df_cy["Timestamp"] <= marker_time_cy + 2)
    seg_cy = df_cy.loc[cy_mask].reset_index(drop=True)

    if seg_cy.empty or seg_rt.empty:
        print("⚠️ 該 marker 資料不足，跳過。")
        continue

    # # 對齊
    min_len = min(len(seg_cy), len(seg_rt))

    # 決定子圖排版（每列最多畫 4 個）
    n_channels = len(channels)
    ncols = 4
    nrows = (n_channels + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 3 * nrows), sharex=True)
    axs = axs.flatten()  # 展平成一維陣列方便處理

    diff_dict = {}
    stats = {}

    for i, ch in enumerate(channels):
        y_c = seg_cy[ch].astype(float).values[:min_len]
        y_r = seg_rt[ch].astype(float).values[:min_len]
        t = seg_cy["Timestamp"].values[:min_len]
        # y_c = apply_notch(y_c, fs)
        # y_r = apply_notch(y_r, fs)
        # y_c = apply_bandpass(y_c, fs, 1, 40)  # 1-40 Hz
        # y_r = apply_bandpass(y_r, fs, 1, 40)
        ax = axs[i]
        diff = overlay_plot(ch, y_c, y_r, t, align=True, n_samples=2000, is_plot=True, ax=ax)
        # diff = overlay_plot(ch, y_c, y_r, t, align=False, is_plot=True, ax=ax)
        diff_dict[ch] = diff

        # 計算統計量
        mask = ~np.isnan(y_c) & ~np.isnan(y_r)
        corr = np.corrcoef(y_c[mask], y_r[mask])[0, 1] if np.sum(mask) >= 2 else np.nan

        stats[ch] = {
            'mean': np.nanmean(diff),
            'std': np.nanstd(diff),
            'rmse': np.sqrt(np.nanmean(diff ** 2)),
            'corr': corr,
        }

    # 統計資料轉 DataFrame 顯示
    stats_df = pd.DataFrame(stats).T
    print("\nChannel 比較統計（cygnus - realtime）:")
    print(stats_df)

    # 移除多餘的子圖
    for j in range(len(channels), len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle("Cygnus vs Real-time Overlay", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])  # 調整空間避免標題擋住子圖
    plt.show()

    # t_cy = seg_cy["Timestamp"].values[:min_len] # plot one by one
    # diff_dict = {}
    # for ch in channels:
    #     y_c = seg_cy[ch].astype(float).values[:min_len]
    #     y_r = seg_rt[ch].astype(float).values[:min_len]
    #     diff = overlay_plot(ch, y_c, y_r, t_cy, align=True, is_plot=False)
    #     # diff = overlay_plot(ch, y_c, y_r, t_cy, align=True, is_plot=True)
    #     diff_dict[ch] = {
    #         "mean": np.nanmean(diff),
    #         "std": np.nanstd(diff),
    #         "rmse": np.sqrt(np.nanmean(diff ** 2)),
    #         "corr": np.corrcoef(y_c, y_r)[0, 1]
    #     }
    #
    # segment_results[marker_label] = pd.DataFrame(diff_dict).T
    # print(segment_results[marker_label])

# ---- 結果輸出 ----
print("\n==== 各 marker 2 秒片段比對結果 ====")
for marker, df_stats in segment_results.items():
    print(f"\nMarker {marker}:")
    print(df_stats)