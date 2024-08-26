@echo off
rem This batch file runs 9 Python scripts in parallel, each using 2 threads and saving outputs in D:\2024_PHD-DATA_CFP_PROCESSING_v2.

rem Set the path to your Python executable if not in PATH
set PYTHON_PATH=C:/Users/jcp/AppData/Local/Programs/Python/Python39/python.exe

rem Set the path to your script
set SCRIPT_PATH=C:\Users\jcp\PycharmProjects\RETFound_MAE_JCP\preprocessing_imgs\per_dir\img_stats_parallel.py


rem Start each process with specified affinity (using 2 threads each)
start "" /affinity 0xC "%PYTHON_PATH%" "%SCRIPT_PATH%" "D:\2024_PHD-DATA_CFP-imgs_cropped\101-120k_240129-1215"
start "" /affinity 0x30 "%PYTHON_PATH%" "%SCRIPT_PATH%" "D:\2024_PHD-DATA_CFP-imgs_cropped\121-140k_240129-2346"
start "" /affinity 0xC0 "%PYTHON_PATH%" "%SCRIPT_PATH%" "D:\2024_PHD-DATA_CFP-imgs_cropped\141-160k_240130-0008"
start "" /affinity 0x300 "%PYTHON_PATH%" "%SCRIPT_PATH%" "D:\2024_PHD-DATA_CFP-imgs_cropped\161-181k_240130-0031"
start "" /affinity 0xC00 "%PYTHON_PATH%" "%SCRIPT_PATH%" "D:\2024_PHD-DATA_CFP-imgs_cropped\1-20k_240128-2147"
start "" /affinity 0x3000 "%PYTHON_PATH%" "%SCRIPT_PATH%" "D:\2024_PHD-DATA_CFP-imgs_cropped\21-40k_240128-2211"
start "" /affinity 0xC000 "%PYTHON_PATH%" "%SCRIPT_PATH%" "D:\2024_PHD-DATA_CFP-imgs_cropped\41-60k_240128-2240"
start "" /affinity 0x30000 "%PYTHON_PATH%" "%SCRIPT_PATH%" "D:\2024_PHD-DATA_CFP-imgs_cropped\61-80k_240128-2307"
start "" /affinity 0xC0000 "%PYTHON_PATH%" "%SCRIPT_PATH%" "D:\2024_PHD-DATA_CFP-imgs_cropped\81-100k_240128-2346"

rem Wait for all started processes to complete
wait

echo All processes have completed.
pause
