import torch


TRAIN_PATH ="data/nyu2_train.csv"
VALIDATION_PATH = "data/nyu2_validation.csv"
TEST_PATH ="data/nyu2_test.csv"

NUM_WORKERS = 0
BATCH_SIZE = 1
NYUD_MEAN = [0.485, 0.456, 0.406]
NYUD_STD = [0.229, 0.224, 0.225]

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 5
MODEL_PATH = 'models/save_model.pt'
LOAD_MODEL = False
TRAIN_MODEL = True
ONLY_MSE_LOSS = False
GRADIENT_FACTOR = 10.
NORMAL_FACTOR = 1.