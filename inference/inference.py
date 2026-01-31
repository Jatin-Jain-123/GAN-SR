import torch
from pathlib import Path
from torch.utils.data import DataLoader
from Scripts.create_dataset import DIV2K
from Scripts.metrics import psnr,ssim,lpips
from models.edsr.src.model.edsr import EDSR
from models.esrgan.RRDBNet_arch import 
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esrgan_psnr,edsr_psnr,esrgan_ssim,edsr_ssim,esrgan_lpips,edsr_lpips=0,0,0,0,0,0

#eval
#no grad
n=data.__len__()
for lr,hr in loader:
    sr_edsr=(lr)
    sr_esrgan=(lr)
    sr_edsr=torch.clamp(sr_edsr,0.0,1.0)
    sr_esrgan=torch.clamp(sr_esrgan,0.0,1.0)
    edsr_psnr=edsr_psnr+psnr(sr_edsr,hr)
    edsr_ssim=edsr_ssim+ssim(sr_edsr,hr)
    edsr_lpips=edsr_lpips+lpips(sr_edsr,hr)
    esrgan_psnr=esrgan_psnr+psnr(sr_esrgan,hr)
    esrgan_ssim=esrgan_ssim+ssim(sr_esrgan,hr)
    esrgan_lpips=esrgan_lpips+lpips(sr_esrgan,hr)
esrgan_psnr,edsr_psnr,esrgan_ssim,edsr_ssim,esrgan_lpips,edsr_lpips=esrgan_psnr/n,edsr_psnr/n,esrgan_ssim/n,edsr_ssim/n,esrgan_lpips/n,edsr_lpips/n
