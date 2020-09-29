import os
from datetime import datetime


#mean and std of cifar10 dataset
CIFAR10_TRAIN_MEAN = [x / 255 for x in [125.3, 123.0, 113.9]]
CIFAR10_TRAIN_STD = [x / 255 for x in [63.0, 62.1, 66.7]]

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 200 
MILESTONES = [60, 120, 160]

#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().isoformat()

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10








