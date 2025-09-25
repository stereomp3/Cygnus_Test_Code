"""
因為 labrecoder 的時間只有提供到秒，所以毫秒需要手動調整 ...
202050923/test 那個多加上 +0.060
"""
import pyxdf
import pandas as pd
import numpy as np
from datetime import datetime

import csv

XDF_FILE = 'data/202050924_signal/SDK_lb/sub-P001_ses-S001_task-Default_run-001_eeg.xdf'
# 儲存結果到 CSV 檔案
csv_file_path = 'data/202050924_signal/SDK_lb/lab_recorder.csv'
# 讀取 XDF 檔案
streams, header = pyxdf.load_xdf(XDF_FILE)

# 提取 EEG 資料流，通常是第一個 stream
eeg_stream = streams[0]
print(header)
print(streams)
# 取得記錄的開始時間
record_datetime_str = header['info']['datetime'][0]
print(record_datetime_str)
record_datetime = datetime.strptime(record_datetime_str, '%Y-%m-%dT%H:%M:%S%z')
record_datetime = record_datetime_str.split('+')[0]  # 去除時區
record_datetime = record_datetime.replace('T', ' ')  # 中間 T 改成空格
# 提取時間戳 (Timestamps) 和 EEG 資料
timestamps = eeg_stream['time_stamps']  # 這是時間戳列表
eeg_data = eeg_stream['time_series']  # 這是所有的 EEG 頻道數據

# 通道名稱 32 channel
channel_names = [
    "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8", "FT7", "FC3", "FCz", "FC4", "FT8",
    "T7", "C3", "Cz", "C4", "T8", "TP7", "CP3", "CPz", "CP4", "TP8", "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2"
]

# 檢查資料的維度，確保 EEG 資料與通道數對應
if eeg_data.shape[1] != len(channel_names):
    raise ValueError(f"EEG 資料與通道數不匹配！預期 {len(channel_names)} 個通道，實際是 {eeg_data.shape[1]} 個通道")

# 創建 DataFrame
df = pd.DataFrame(eeg_data, columns=channel_names)

# 計算每個時間戳的具體時間（Timestamp）
# 假設時間戳是從某個基準時間開始的
time_in_seconds = np.arange(0, len(timestamps) * 0.002, 0.002)  # 每 0.02 秒累加
formatted_time = [f"{x:.3f}" for x in time_in_seconds]  # 轉換成格式化的時間戳

# 將時間戳加到 DataFrame
df['Timestamp'] = formatted_time

# 重排欄位順序，將 Timestamp 放到最前面
df = df[['Timestamp'] + channel_names]

# 四捨五入 EEG 資料至小數點後兩位
for col in channel_names:
    df[col] = df[col].apply(lambda x: round(x, 2))  # 四捨五入至小數點後兩位
# 開始寫 CSV 檔案，將 Record datetime 寫到第一行
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([f"Record datetime: {record_datetime}"])  # 第一行寫入 Record datetime
    # 然後寫入欄位名稱（標題列）
    writer.writerow(['Timestamp'] + channel_names)
    # 寫入 EEG 資料
    for row in df.itertuples(index=False, name=None):
        writer.writerow(row)

print("轉換完成，已儲存為 lab_recorder.csv")

