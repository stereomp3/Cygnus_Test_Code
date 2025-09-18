# compare_with_csv_no_merge.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from scipy.signal import correlate, iirnotch, butter, filtfilt
from sklearn.linear_model import LinearRegression


# ---- helper functions ----
fs = 500  # 500 Hz
def design_notch(fs, freq=50, Q=30):
    """設計 notch 濾波器"""
    b, a = iirnotch(freq / (fs / 2), Q)
    return b, a


def apply_notch(data, fs, freq=50, Q=30):
    """應用 notch 濾波器於數據"""
    b, a = design_notch(fs, freq, Q)
    return filtfilt(b, a, data, axis=0)


def apply_bandpass(data, fs, low, high, order=4):
    """應用帶通濾波器於數據"""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

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
    """Return lag in samples for best alignment (y_r shifted to match y_c)."""
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
# cygnus_path = "test_data/_cygnus_record.csv" # 沒腦波資料
# realtime_path = "test_data/_realtime_record.csv"

cygnus_path = "data/250914175336.csv"  # 腦波資料
realtime_path = "data/eeg_record.csv"

df_cy, dt_cy = read_cygnus_csv(cygnus_path)
df_rt, dt_rt = read_realtime_csv(realtime_path)

# 計算 Record datetime 差異
time_diff = dt_cy - dt_rt
print("Record datetime 差（cygnus - realtime）:", time_diff)

# drop unnamed/empty columns
df_cy = df_cy.loc[:, ~df_cy.columns.str.contains('^Unnamed')]
df_rt = df_rt.loc[:, ~df_rt.columns.str.contains('^Unnamed')]

# 轉成 numeric timestamp
df_cy['Timestamp'] = pd.to_numeric(df_cy['Timestamp'], errors='coerce')
df_rt['Timestamp'] = pd.to_numeric(df_rt['Timestamp'], errors='coerce')
df_cy = df_cy.dropna(subset=['Timestamp']).reset_index(drop=True)
df_rt = df_rt.dropna(subset=['Timestamp']).reset_index(drop=True)

# 建立絕對時間欄位
df_cy['abs_time'] = dt_cy + pd.to_timedelta(df_cy['Timestamp'], unit='s')
df_rt['abs_time'] = dt_rt + pd.to_timedelta(df_rt['Timestamp'], unit='s')

# 找出共同 channels（排除 metadata）
exclude = {'Timestamp', 'Serial Number', 'Event Id', 'Event Date', 'Event Duration',
           'Software Marker', 'Software Marker Name', 'abs_time'}
channels = [c for c in df_cy.columns if c in df_rt.columns and c not in exclude]
print("偵測到要比對的 channels:", channels)

# # 依照 abs_time 進行對齊(取 2 秒內最接近的資料)
# df_cy_sorted = df_cy.sort_values('abs_time')
# df_rt_sorted = df_rt.sort_values('abs_time')

# 使用 merge_asof 對齊時間(以 cygnus 為基準)
df_merged = pd.merge_asof(
    df_cy,
    df_rt,
    on='abs_time',
    direction='nearest',
    tolerance=pd.Timedelta(seconds=2),
    suffixes=('_cy', '_rt')
)

# 移除沒有成功對齊的資料
df_merged = df_merged.dropna(subset=[f'{ch}_rt' for ch in channels])

print(f"成功對齊的資料筆數：{len(df_merged)}")
# 決定子圖排版（每列最多畫 4 個）
n_channels = len(channels)
ncols = 4
nrows = (n_channels + ncols - 1) // ncols

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 3 * nrows), sharex=True)
axs = axs.flatten()  # 展平成一維陣列方便處理

diff_dict = {}
stats = {}

for i, ch in enumerate(channels):
    y_c = df_merged[f'{ch}_cy'].to_numpy()
    y_r = df_merged[f'{ch}_rt'].to_numpy()
    t = (df_merged['abs_time'] - df_merged['abs_time'].iloc[0]).dt.total_seconds()
    # y_c = apply_notch(y_c, fs)
    # y_r = apply_notch(y_r, fs)
    # y_c = apply_bandpass(y_c, fs, 1, 40)  # 1-40 Hz
    # y_r = apply_bandpass(y_r, fs, 1, 40)
    ax = axs[i]
    diff = overlay_plot(ch, y_c, y_r, t, align=False, n_samples=2000, is_plot=True, ax=ax)
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


