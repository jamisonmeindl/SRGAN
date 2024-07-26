import os
import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

def calculate_metrics(image1_path, image2_path):
    img1 = io.imread(image1_path)
    img2 = io.imread(image2_path)
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions.")
    ssim_value = ssim(img1, img2, channel_axis=-1, data_range=img2.max() - img2.min())
    psnr_value = psnr(img1, img2, data_range=img2.max() - img2.min())
    return ssim_value, psnr_value

def process_folders(folder_path1, folder_path2):
    directories1 = [d for d in os.listdir(folder_path1) if os.path.isdir(os.path.join(folder_path1, d))]
    directories2 = [d for d in os.listdir(folder_path2) if os.path.isdir(os.path.join(folder_path2, d))]
    common_directories = set(directories1).intersection(directories2)

    all_ssim = []
    all_psnr = []
    
    for directory in common_directories:
        dir1 = os.path.join(folder_path1, directory)
        dir2 = os.path.join(folder_path2, directory)
        images1 = {f: os.path.join(dir1, f) for f in os.listdir(dir1) if f.endswith(('.png', '.jpg'))}
        images2 = {f: os.path.join(dir2, f) for f in os.listdir(dir2) if f.endswith(('.png', '.jpg'))}
        
        common_images = set(images1.keys()).intersection(images2.keys())
       
        for image in common_images:
            ssim_value, psnr_value = calculate_metrics(images1[image], images2[image])
            all_ssim.append(ssim_value)
            all_psnr.append(psnr_value)
    
    mean_ssim = np.mean(all_ssim) if all_ssim else 0
    mean_psnr = np.mean(all_psnr) if all_psnr else 0
    
    return mean_ssim, mean_psnr

# Example usage
folder_path1 = 'data/REDS/val_REDS4_sharp_bicubic/X4'
folder_path2 = 'data/REDS/base_pred_val'

# folder_path1 = 'data/VID4/GT'
# folder_path2 = 'data/VID4/BDx4/base_pred'

folder_path1 = 'data/REDS/val_REDS4_sharp'
folder_path2 = 'data/REDS/base_highres_pred_val'

mean_ssim, mean_psnr = process_folders(folder_path1, folder_path2)
print(f"Mean SSIM: {mean_ssim}")
print(f"Mean PSNR: {mean_psnr}")
