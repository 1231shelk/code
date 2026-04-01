import os
import torch

BATCH_SIZE = 64  # 64
EPOCHS = 150
LR = 3e-5
GAMMA = 0.7
SEED = 42

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_PATH = r'./'
TRAIN_DIR = os.path.join(DATA_PATH, 'E:/李志杰/小鼠卵泡病变识别/AP_VCT/data/MultiDistancePhaTrain')
VALID_DIR = os.path.join(DATA_PATH, 'E:/李志杰/小鼠卵泡病变识别/AP_VCT/data/MultiDistancePhaValid')
TEST_DIR = os.path.join(DATA_PATH, 'E:/李志杰/小鼠卵泡病变识别/AP_VCT/data/MultiDistancePhaTest')
MODEL_SAVE_DIR = os.path.join(DATA_PATH, 'D:/黄骁文/VITHolo/PhaRes/model')
CSV_SAVE_DIR = os.path.join(DATA_PATH, 'D:/黄骁文/VITHolo/PhaRes/result/curve')
MATRIX_SAVE_DIR = os.path.join(DATA_PATH, 'D:/黄骁文/VITHolo/PhaRes/result/matrix')


