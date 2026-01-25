import torch
from pathlib import Path
from torch.utils.data import DataLoader
from Scripts.create_dataset import DIV2K
SEED=42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
root = Path(__file__).resolve().parent.parent/"dataset"/"val"
lr_dir=root/"LR_x4"
hr_dir=root/"HR"
data=DIV2K(lr_dir,hr_dir,"eval")
loader=DataLoader(data,batch_size=1,shuffle=False,num_workers=0,pin_memory=False)
lr,hr = next(iter(loader))
print(lr.shape,hr.shape)
