import os

cwd = list(os.getcwd().split(os.sep))
STIMA_dir = os.sep.join(cwd[:len(cwd)-2])

ACam_path = os.path.join(STIMA_dir, 'ACam')
database_path = os.path.join(STIMA_dir, 'bulb_database', 'bulb_database_master.csv')
base_path = os.path.join(STIMA_dir, 'bulb_database', 'csv_files')
gradient_save_path = os.path.join(STIMA_dir, 'Gradient Tests', 'Savgol 31 Moving 50')
raw_waveform_save_path = os.path.join(STIMA_dir, 'bulb_database', 'Raw BRFs')

brf_analysis_save_path = os.path.join(STIMA_dir, 'BRF_Analysis')
feature_analysis_save_directory = os.path.join(STIMA_dir, 'BRF_Analysis', 'Feature Analysis')

file_renaming_path = os.path.join(ACam_path, 'LIVE', 'file_renaming')
consolidated_folder_path = os.path.join(ACam_path, 'LIVE', 'Outdoor_Testing')

# ACam_db path
ACam_db_path = os.path.join(STIMA_dir, 'ACam', 'ACam_Training_Data')

# "global" variables
savgol_window = 31
mov_avg_w_size = 50