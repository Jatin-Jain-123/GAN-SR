import os
from glob import glob
# Alternative - from pathlib import Path
import cv2
scale=4
hr_dir_train = "../dataset/train/HR"
lr_dir_train = "../dataset/train/LR_x4"
hr_dir_val = "../dataset/val/HR"
lr_dir_val = "../dataset/val/LR_x4"
os.makedirs(lr_dir_train,exist_ok=True)
os.makedirs(lr_dir_val,exist_ok=True)
# alternative - for img_path in Path(hr_dir_train).glob("*.png"):
for img_path in glob(os.path.join(hr_dir_train,"*.png")):
    img = cv2.imread(img_path)
    h,w,_ = img.shape
    lr_img = cv2.resize(img, (w//scale, h//scale), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(lr_dir_train,os.path.basename(img_path)),lr_img)
# alternative - for img_path in Path(hr_dir_val).glob("*.png"):
for img_path in glob(os.path.join(hr_dir_val,"*.png")):
    img = cv2.imread(img_path)
    h,w,_=img.shape
    lr_img=cv2.resize(img,(w//scale,h//scale),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(lr_dir_val,os.path.basename(img_path)),lr_img)