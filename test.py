from src.dataset import Dataset
from src.preprocess import Preprocess

# preprocessor = Preprocess(raw_data_path='./data/raw', processed_data_path='./data/processed')
# preprocessor.preprocess_s1()
# preprocessor.preprocess_l8_9()
# preprocessor.run()
dataset_L8_9 = Dataset(processed_data="./data/processed", sensor_type="L8_9")
dataset_L8_9.invetory
dataset_s2 = Dataset(processed_data="./data/processed", sensor_type="S2")
# dataset_L8_9.create_pactches()
# dataset_s2.create_pactches()


# for i in range(len(dataset_L8_9)):
#     data = dataset_L8_9[i][0]
#     print(f"L8 Data {i} shape: {data.shape}")

# for i in range(len(dataset_s2)):
#     data = dataset_s2[i][0]
#     print(f"S2 Data {i} shape: {data.shape}")
#     print(data[:,:,-1].max())
