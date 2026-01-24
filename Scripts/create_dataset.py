import torch
import cv2
from torch.utils.data import Dataset
from pathlib import Path
class DIV2K(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir=Path(lr_dir)
        self.hr_dir=Path(hr_dir)
        self.filenames = sorted(set(p.name for p in self.hr_dir.iterdir()) & set(p.name for p in self.lr_dir.iterdir()))
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self,idx):
        name = self.filenames[idx]
        lr=self.lr_dir/name
        hr=self.hr_dir/name
        lr=cv2.imread(str(lr))
        hr=cv2.imread(str(hr))
        if lr is None or hr is None:
            raise RuntimeError(f"Failed to load image: {name}")
        assert hr.shape[0]==lr.shape[0]*4
        assert hr.shape[1]==lr.shape[1]*4
        # use assert statements during development only, replace them with if-else statements during production/deployment
        """ if hr.shape[0]!=lr.shape[0]*4 or hr.shape[1]!=lr.shape[1]*4:
                raise ValueError("Expected HR to be 4× LR in height and width")
        """
        lr=cv2.cvtColor(lr,cv2.COLOR_BGR2RGB)
        hr=cv2.cvtColor(hr,cv2.COLOR_BGR2RGB)
        lr=torch.from_numpy(lr)
        lr=lr.permute(2,0,1)
        lr=lr.float()
        lr=lr/255.0
        hr=torch.from_numpy(hr).permute(2,0,1).float()/255.0
        return lr,hr
