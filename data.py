import random
import pandas as pd
from parameters import TRAIN_PATH

train_df = pd.read_csv(TRAIN_PATH)
train_paths = train_df.values.tolist()
random.shuffle(train_paths)
train_info = train_paths[:40000]
val_info = train_paths[40000:]
train_ = pd.DataFrame(train_info)
val_ = pd.DataFrame(val_info)
train_.to_csv('data/nyu2_train.csv', index=False, header=False)
val_.to_csv('data/nyu2_validation.csv', index=False, header=False)

