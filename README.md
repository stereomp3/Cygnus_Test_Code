# Cygnus_Test_Code
`compare_all_plot.py`: 讀取 lsl real time 的 eeg 資料，與 cygnus 錄製的資料進行比對，會先把資料的時間轉換成絕對時間，然後使用相同的時間去做資料比對，如果設定 overlay_plot(..., align=True)，會計算出最佳延遲，然後做後續的計算。最後會給出每個 channel 相差的 mean, std, rmse, corr。後來發現 "計算出最佳延遲" 會有些不準確，所以改使用其他方式驗證。

`lsl_eeg_recoder.py`: 使用 LSL 接收 eeg 腦波帽的訊號，並寫入到 csv 檔案裡面，在最開始會記錄 `Time.time()`，利用這個資訊可以與 cygnus 時間對齊。在最開始會創建 Marker stream，並會送出 marker，以便後續測試

`lsl_eeg_recoder_pull_chunk.py`: 使用 pull chunk 的方式，與 lsl_eeg_recoder 基本一樣。



程式執行步驟: 執行 `lsl_eeg_recoder.py` -> cygnus 連線 LSL Marker stream -> cygnus 錄製 -> wait for a while -> cygnus 結束錄製 -> cygnus 結束 LSL Marker stream 連線 -> 結束 `lsl_eeg_recoder.py`



其他程式執行步驟 執行 `錄製 LSL 資料的 python 檔案` -> labrecoder 錄製 -> cygnus 錄製   -> wait for a while -> cygnus 結束錄製 -> labrecoder 結束錄製 -> 結束 `錄製 LSL 資料的 python 檔案`
