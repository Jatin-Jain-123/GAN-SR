import torch
import torch.nn.functional as F
import lpips
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
def psnr(sr,hr,max_value=1.0):
    mse = F.mse_loss(sr,hr)
    eps = 1e-10
    return 20*torch.log10(max_value/torch.sqrt(mse+eps))
ssim_metric=StructuralSimilarityIndexMeasure(data_range=1.0)
def ssim(sr, hr):
    return ssim_metric(sr,hr)
lpips_metric=lpips.LPIPS(net="vgg")
def lpips(sr,hr):
    sr=sr*2-1
    hr=hr*2-1
    return lpips_metric(sr,hr).mean()