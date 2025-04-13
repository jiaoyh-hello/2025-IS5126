# Import Auxiliary modules
import pandas as pd
import os

# Set the loading path
proj_dir = r"E:\NUS\IS5126\Module BERT\MachineLearningProject\Data\raw"
train_file_path = os.path.join(proj_dir, "train.csv")

# load raw data
def load_data(data_file_path):
    data = pd.read_csv(data_file_path)
    print("Shape of data:", data.shape)
    print("Data preview:")
    print(data.head())
    return data

if __name__ == '__main__':
    # Assuming train.csv and test.csv are located in the current working directory
    train_data, test_data = load_data(train_file_path)