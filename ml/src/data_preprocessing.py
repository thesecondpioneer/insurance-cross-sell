import pandas as pd
import numpy as np
from pathlib import Path

# Определяем пути
current_dir = Path(__file__).parent  # ml/src/
ml_dir = current_dir.parent

input_path = ml_dir / 'data' / 'raw'/'train.csv'
output_path = ml_dir / 'data' / 'processed' / 'train_processed.csv'
data = pd.read_csv(input_path, nrows = 5000000, index_col=0)

data.to_csv(output_path, index=False)