from pathlib import Path
from torch.utils.data import DataLoader
from Scripts.create_dataset import DIV2K
root = Path(__file__).resolve().parent.parent/"dataset"/"val"
lr_dir=root/"LR_x4"
hr_dir=root/"HR"
data=DIV2K(lr_dir,hr_dir,"eval")
loader=DataLoader(data,batch_size=4,shuffle=False,num_workers=2,pin_memory=False)
lr,hr = next(iter(loader))
print(lr.shape,hr.shape)
