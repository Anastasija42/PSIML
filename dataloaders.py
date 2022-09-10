import random
from torch.utils.data import DataLoader
from dataset import MyDataset
import pandas as pd
from parameters import TRAIN_PATH, VALIDATION_PATH, TEST_PATH, NUM_WORKERS, BATCH_SIZE


train_df = pd.read_csv(TRAIN_PATH)
train_info = train_df.values.tolist()
val_df = pd.read_csv(VALIDATION_PATH)
val_info = val_df.values.tolist()
test_df = pd.read_csv(TEST_PATH)
test_info = test_df.values.tolist()


train_set = MyDataset(train_info, transform=True)
val_set = MyDataset(val_info)
test_set = MyDataset(test_info)

train_dataloader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
valid_dataloader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
test_dataloader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)