import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from Scripts.create_dataset import DIV2K 
root=Path(__file__).resolve().parent.parent/"dataset"/"train"
lr_dir=root/"LR_x4"
hr_dir=root/"HR"
data=DIV2K(lr_dir,hr_dir,"train")
loader=DataLoader(data,batch_size=4,shuffle=True,num_workers=0,pin_memory=False)
lr,hr = next(iter(loader))
print(lr.shape,hr.shape)