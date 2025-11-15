import pandas as pd
import numpy as np

file_path = 'C:/PyCharmProjects/Git_ML/ml/data/raw/train.csv'  # изменить путь на нужный
data = pd.read_csv(file_path, nrows = 5000000, index_col=0)

