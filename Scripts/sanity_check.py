from create_dataset import DIV2K
from pathlib import Path
path = Path(__file__).resolve().parent.parent
path = path/"dataset"/"val"
hr_path = path/"HR"
lr_path = path/"LR_x4"
ds = DIV2K(lr_path,hr_path)
lr, hr = ds[0]
print(lr.shape, hr.shape)
print(lr.dtype,hr.dtype)
