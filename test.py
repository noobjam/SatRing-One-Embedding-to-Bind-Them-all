from src.preprocess import Preprocess

preprocessor = Preprocess(raw_data_path='./data/raw', processed_data_path='./data/processed')
preprocessor.preprocess_s2()
# meta =preprocessor._get_metadata('data/raw/S2/S2A_MSIL2A_20240808T021341_N0511_R060_T51PZL_20240808T081405.SAFE/GRANULE/L2A_T51PZL_A047676_20240808T021344/IMG_DATA/R10m/T51PZL_20240808T021341_B02_10m.jp2')
# print(meta)