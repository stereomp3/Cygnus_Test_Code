"""
單純改 pull chunk，改 read_eeg 的部分，其他差不多
"""
import time
import csv
import pickle
import threading
import warnings
import numpy as np
from collections import deque, Counter
from pylsl import StreamInfo, StreamOutlet, resolve_byprop, StreamInlet, resolve_streams
from scipy.signal import iirnotch, filtfilt, butter, detrend, welch
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
import sys
from datetime import datetime
from functools import wraps
import os

# ----------------------------
# Parameters (tweak as needed)
# ----------------------------
STREAM_NAME = "Cygnus-329018-RawEEG"
# FS = 500
N_CHANNELS = 32  # 32 channel eeg cap
SAVE_CSV = True
BASE_FILE = "data/202050924_signal/chunk/chunk_"
CSV_FILENAME = f"{BASE_FILE}eeg_record.csv"

csv_file = None
csv_writer = None


# 自動儲存 log
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def tee_log(log_file=None):
    """裝飾器：將 print 輸出同時存檔與顯示"""
    if log_file is None:
        log_file = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout
            with open(log_file, "w", encoding="utf-8") as f:
                sys.stdout = Tee(original_stdout, f)
                try:
                    result = func(*args, **kwargs)
                finally:
                    sys.stdout = original_stdout
            print(f"✅ 輸出已保存到 {log_file}")
            return result

        return wrapper

    return decorator


# ----------------------------
# LSL Marker stream (outlet)
# ----------------------------
def setup_marker_stream():
    info = StreamInfo('MarkerStream', 'Markers', 1, 0, 'int32', 'marker_stream_id')
    outlet = StreamOutlet(info)
    return outlet


marker_outlet = setup_marker_stream()
MARKER_CSV = f"{BASE_FILE}markers.csv"
ts_lsl = None
with open(MARKER_CSV, "w", newline="") as f:  # 初始化文件
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Marker label"])


def send_marker(label: int):
    """Send integer marker through LSL, update latest_marker, and log to CSV."""
    global ts_lsl
    # print(f"ts_lsl: {ts_lsl}")
    # 1. push to LSL marker outlet
    try:
        marker_outlet.push_sample([int(label)])
    except Exception as e:
        print("[Marker] Push failed:", e)

    # 2. save to marker CSV
    with open(MARKER_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts_lsl, int(label)])

    print(f"[Marker] Sent: {label}")


# ----------------------------
# LSL inlet setup
# ----------------------------
def setup_lsl_inlet(stream_name=STREAM_NAME, timeout=5.0):
    print("Resolving streams...")
    streams = resolve_streams()
    for s in streams:
        try:
            print("Found stream:", s.name())
        except Exception:
            pass
    streams = resolve_byprop('name', stream_name, timeout=timeout)
    if len(streams) == 0:
        raise RuntimeError(f"Could not resolve LSL stream: {stream_name}")
    inlet = StreamInlet(streams[0])
    print("Connected to LSL stream:", streams[0].name())
    return inlet


# ----------------------------
# Read EEG continuously (thread)
# ----------------------------
def read_eeg(inlet, save_csv=SAVE_CSV, filename=CSV_FILENAME, chunk_size=16):
    """
    Continuously pull chunks of samples from inlet and append to eeg_buffer.
    Optionally write each sample + timestamp + marker to CSV.
    """
    global eeg_buffer, csv_file, csv_writer, ts_lsl

    if save_csv:
        csv_file = open(filename, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([f"Record datetime: {datetime.now()}"])
        header = ["Timestamp"] + [
            "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
            "FT7", "FC3", "FCz", "FC4", "FT8", "T7", "C3", "Cz", "C4", "T8",
            "TP7", "CP3", "CPz", "CP4", "TP8", "P7", "P3", "Pz", "P4", "P8",
            "O1", "Oz", "O2"
        ]
        csv_writer.writerow(header)

    first_ts_lsl = None
    while True:
        # Pull a chunk of samples
        chunk, ts = inlet.pull_chunk(timeout=0.0, max_samples=chunk_size)  # Change here: pull a chunk of samples

        if len(chunk) == 0:
            continue

        if first_ts_lsl is None:
            first_ts_lsl = ts[0]  # Initialize the first timestamp if not already set

        # Process each sample in the chunk
        for sample, ts_item in zip(chunk, ts):
            arr = np.array(sample).reshape(-1)
            arr = np.around(arr, decimals=2)  # Round values to 2 decimal places

            # Ensure correct channel count
            if arr.size != N_CHANNELS:
                if arr.size < N_CHANNELS:
                    pad = np.zeros(N_CHANNELS - arr.size)
                    arr = np.concatenate([arr, pad])
                else:
                    arr = arr[:N_CHANNELS]

            # Write CSV row if enabled
            if save_csv and csv_writer is not None:
                try:
                    ts_lsl = f"{ts_item - first_ts_lsl:.3f}"  # 使用 ts_item（來自 zip(chunk, ts)）來計算時間差
                    row = [ts_lsl] + arr.tolist()
                    csv_writer.writerow(row)
                except Exception as e:
                    print("[CSV] write error:", e)


# ----------------------------
# Main flow
# ----------------------------
@tee_log(f"{BASE_FILE}log_20250903.txt")
def main_flow():
    global csv_file, csv_writer

    inlet = setup_lsl_inlet(STREAM_NAME)

    # start EEG reading thread
    reader_thread = threading.Thread(target=read_eeg, args=(inlet, SAVE_CSV, CSV_FILENAME), daemon=True)
    reader_thread.start()

    input("wait for input, and input for start...")
    # give reader some time to fill buffer
    print("[Main] Waiting briefly for buffer to fill...")
    send_marker(13)  # for test start
    time.sleep(1.0)

    # Step1: baseline
    print("[Main] Baseline starting in 2s...")
    send_marker(12)  # for test
    time.sleep(2.0)
    send_marker(10)  # for test
    time.sleep(4)
    send_marker(11)  # baseline end
    input("wait for input, and input for end...")
    # close CSV
    if csv_file:
        try:
            csv_file.close()
            print("[CSV] Saved EEG data to", CSV_FILENAME)
        except Exception:
            pass



if __name__ == "__main__":
    try:
        main_flow()
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting.")
        if csv_file:
            try:
                csv_file.close()
            except Exception:
                pass
